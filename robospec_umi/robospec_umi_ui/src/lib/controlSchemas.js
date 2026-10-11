/**
 * What sliders each camera offers, and what to say about them.
 *
 * The panel is schema-driven so one component serves both cameras. Everything
 * about WHICH controls exist and what they mean lives here; ControlPanel only
 * knows how to render a group.
 *
 * `disabled` is never derived here — it comes from `ctrl.adjustable`, which both
 * backends already compute. The server is the authority on what is writable.
 */

export const SCENE_SCHEMA = {
  brightnessAxes: ['exposure_time_absolute', 'gain'],
  groups: [
    {
      // Gain lives in the exposure group because auto exposure drives BOTH --
      // it is a brightness loop, and raising gain just makes it shorten the
      // exposure to land on the same target.
      toggle: {
        name: 'auto_exposure', label: 'Auto exposure', on: 3, off: 1,
        hint: 'drives exposure time and gain together',
      },
      sliders: [
        {
          name: 'exposure_time_absolute', label: 'Exposure',
          hint: '800 is one frame period at 120 fps — the sensor cannot integrate longer, so higher only clips',
          hintWhenAuto: 'the camera is driving this',
        },
        {
          name: 'gain', label: 'Gain',
          hint: 'flat above ~250 on this sensor',
          hintWhenAuto: 'auto exposure drives gain as well as exposure',
        },
      ],
    },
    {
      toggle: {
        name: 'white_balance_automatic', label: 'Auto white balance',
        on: 1, off: 0,
      },
      sliders: [{ name: 'white_balance_temperature', label: 'White balance (K)' }],
    },
    {
      sliders: [{
        name: 'gamma', label: 'Gamma',
        hint: 'a display curve applied after digitisation — it still works under auto, but prefer light over gamma',
      }],
    },
  ],
}

const WRIST_D405 = {
  brightnessAxes: ['exposure', 'gain'],
  // Measured, not assumed: the D405's auto exposure pins gain at the sensor
  // floor (16) and leans on integration time alone, landing at mean luma ~10
  // against ~62 for manual 6000us/gain 248. It is tuned for the stereo depth
  // pipeline, where a short dark frame is what you want.
  banner: 'No auto exposure on this camera: it pins gain at the floor (16) and '
        + 'produces a near-black image. Exposure and gain stay manual.',
  groups: [
    {
      sliders: [
        {
          name: 'exposure', label: 'Exposure (µs)',
          hint: 'capped at 10000 — past the 11111 µs frame period the camera silently drops to 60 fps, with no dropped frames to notice',
        },
        {
          name: 'gain', label: 'Gain',
          hint: 'floor is 16, not 0 — and it is the free axis here: no blur, no frame-rate cost, no clipping even at 248',
        },
      ],
    },
    {
      toggle: {
        name: 'enable_auto_white_balance', label: 'Auto white balance',
        on: 1, off: 0,
        hint: 'no frame-rate or brightness cost, but the camera will not report the value it picks',
      },
      sliders: [{ name: 'white_balance', label: 'White balance (K)' }],
    },
    {
      sliders: [{ name: 'gamma', label: 'Gamma' }],
    },
  ],
}

// Measured on this unit (d455_test/d455_documentation.md). Auto exposure runs with
// auto_exposure_priority pinned to 0, so it never trades frame rate for light.
const WRIST_D455 = {
  brightnessAxes: ['exposure', 'gain'],
  groups: [
    {
      toggle: {
        name: 'enable_auto_exposure', label: 'Auto exposure', on: 1, off: 0,
        hint: 'holds 90 fps: it shortens exposure and raises gain, never lowers the frame rate',
      },
      sliders: [
        {
          name: 'exposure', label: 'Exposure (µs)',
          hint: 'capped at 10900 µs: 11.0–11.1 ms gives black frames, and longer lowers the frame rate',
          hintWhenAuto: 'auto exposure is driving this; the camera does not report the value it picks',
        },
        {
          name: 'gain', label: 'Gain',
          hint: '0–128 on the D455',
          hintWhenAuto: 'auto exposure drives gain as well',
        },
      ],
    },
    {
      toggle: {
        name: 'enable_auto_white_balance', label: 'Auto white balance',
        on: 1, off: 0,
        hint: 'no frame-rate or brightness cost, but the camera will not report the value it picks',
      },
      sliders: [{ name: 'white_balance', label: 'White balance (K)' }],
    },
    {
      sliders: [{ name: 'gamma', label: 'Gamma' }],
    },
  ],
}

// The wrist schema follows the camera the server reports (controls._model).
export const WRIST_SCHEMAS = { D405: WRIST_D405, D455: WRIST_D455 }

export const SCHEMAS = { scene: SCENE_SCHEMA, wrist: WRIST_D405 }
