# robospec_umi: calibration, capture, dataset build, training and evaluation in
# one image, plus the server that will host the web UI.
#
# Build context is the REPO ROOT, not robospec_umi/ -- diffusion_policy/ and umi/
# live there and build_zarr.py imports both.
#
#   docker compose -f robospec_umi/robospec_umi_compose.yaml build
#
# Single stage on purpose. A node stage for the UI would be the natural second
# stage, but robospec_umi_ui/ has no package.json yet, so `npm ci` would fail the
# build. Until then the UI directory is bind-mounted and served live; add the
# node stage when there is something to compile.

FROM mambaorg/micromamba:1.5-jammy
USER root

# The base image's entrypoint activates an env before running the command. Not
# needed here -- PATH points at the env's bin directly (see below) -- and it gets
# in the way of `docker compose exec`, so it is cleared.
ENTRYPOINT []

ENV NVIDIA_VISIBLE_DEVICES=all \
    # `video` is the capability people miss. The toolkit's default is
    # compute,utility, which mounts /dev/nvidia* but NOT libnvidia-encode -- and
    # h264_nvenc is WristRecorder's default encoder, so capture fails at runtime
    # with an error that points nowhere near the container configuration.
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    PYTHONUNBUFFERED=1 \
    # cv2.imshow through an X11 socket: MIT-SHM is a same-host optimisation that
    # does not survive the container boundary and shows up as a blank window.
    QT_X11_NO_MITSHM=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      # capture.py and calibrate_scene_cam.py shell out to v4l2-ctl
      v4l-utils \
      # librealsense: USB access, device resets, udev lookups
      libusb-1.0-0 udev \
      # conda-forge's opencv is a Qt build; these are what cv2.imshow needs
      libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 \
      libxkbcommon-x11-0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 \
      libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
      libdbus-1-3 libfontconfig1 \
      # the conda yaml's pip section needs these to BUILD, not just to run:
      #   spnav           -> libspnav-dev, spacenavd
      #   free-mujoco-py  -> libosmesa6-dev, libglfw3, patchelf
      libspnav-dev spacenavd libosmesa6-dev libglfw3 patchelf \
      # ur-rtde and anything else without a wheel
      build-essential cmake \
      # curl for healthchecks, procps so ps/top work when debugging in here
      curl procps ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------- conda env
# The same yaml the host uses, unmodified, so the container and the host are the
# same environment. micromamba rather than full conda: identical result off the
# same file, about a gigabyte smaller and a much faster solve.
ENV MAMBA_ROOT_PREFIX=/opt/conda \
    # pip section of the yaml: tolerate slow PyPI downloads
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10
COPY robospec_umi/robospec_umi_conda.yaml /tmp/env.yaml
# The cache purge must be in THIS layer: a later `rm` cannot shrink an image,
# it only adds a whiteout. Worth ~800 MB -- less than `du` on pkgs/ suggests,
# because conda hardlinks package files into the env rather than copying them.
RUN micromamba create -y -f /tmp/env.yaml && \
    micromamba clean -a -y && \
    rm -rf /opt/conda/pkgs/* /root/.cache && \
    rm /tmp/env.yaml

# Activation by PATH instead of an entrypoint wrapper: every RUN, CMD and
# `compose exec` gets the env without needing a login shell.
ENV PATH=/opt/conda/envs/robospec_umi/bin:$PATH

# mujoco_py cythonises cymj.pyx on FIRST import and writes the result into
# site-packages. Done here as root it is baked into the image; left until runtime
# it fails, because by then the process is the unprivileged robospec user and
# /opt/conda is not writable. Nothing in this pipeline imports it -- this is here
# so the image really can run everything the env claims to provide.
RUN python -c "import mujoco_py" || \
    echo "WARNING: mujoco_py did not build; robosuite/mujoco features unavailable"

# ------------------------------------------------------------------- user
# Files written into the bind-mounted repo must be owned by the host user, not
# root. UID/GID default to 1000 (the usual first human account); override at
# build time if yours differ -- see the README.
#
# No collision with the base image's mambauser, which sits at UID 57439.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} robospec && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash robospec && \
    usermod -aG video,plugdev,audio robospec && \
    # mujoco_py takes a build lock in its own package directory on every import,
    # not just the first, so pre-compiling above is not enough -- that directory
    # has to stay writable by the user actually running it.
    chown -R ${UID}:${GID} \
      /opt/conda/envs/robospec_umi/lib/python3.9/site-packages/mujoco_py/generated \
      2>/dev/null || true

USER robospec
WORKDIR /workspace

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD curl -sf http://127.0.0.1:8080/health || exit 1

CMD ["python", "robospec_umi/robospec_umi_server.py", \
     "--host", "0.0.0.0", "--port", "8080"]
