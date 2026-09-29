import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Home from '@/pages/Home'
import Calibration from '@/pages/Calibration'
import CalibrationNew from '@/pages/CalibrationNew'
import CalibrationTest from '@/pages/CalibrationTest'
import CaptureNew from '@/pages/CaptureNew'
import Edit from '@/pages/Edit'
import EditProject from '@/pages/EditProject'
import Placeholder from '@/pages/Placeholder'
import Training from '@/pages/Training'

// Real routes rather than a useState switch: a calibration capture session lives
// on the server, so a refresh mid-run should land you back on the same stage
// rather than at the front door.
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/calibration" element={<Calibration />} />
        <Route path="/calibration/test" element={<CalibrationTest />} />
        <Route path="/calibration/new" element={<Navigate to="/calibration/new/lock" replace />} />
        <Route path="/calibration/new/:stage" element={<CalibrationNew />} />

        <Route path="/capture" element={<Navigate to="/capture/new/lock" replace />} />
        <Route path="/capture/new" element={<Navigate to="/capture/new/lock" replace />} />
        <Route path="/capture/new/:stage" element={<CaptureNew />} />
        <Route path="/edit" element={<Edit />} />
        <Route path="/edit/:pid" element={<Navigate to="select" replace />} />
        <Route path="/edit/:pid/:stage" element={<EditProject />} />
        <Route path="/training" element={<Training />} />
        <Route path="/evaluation" element={
          <Placeholder title="Evaluation"
            blurb="Score a checkpoint against recorded data, with no robot attached."
            cli={'python robospec_umi/robospec_umi_evaluation/eval_without_robot.py'} />} />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
