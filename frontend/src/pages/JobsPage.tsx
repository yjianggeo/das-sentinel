import { useCallback, useState } from "react";
import JobCompletionEstimate from "../components/JobCompletionEstimate";
import ModelScans from "../components/ModelScans";
import WindowJobs from "../components/WindowJobs";
import DirectoryScans from "../components/DirectoryScans";
import ScanCoverageCalendar from "../components/ScanCoverageCalendar";

export default function JobsPage() {
  const [windowJobsRevision, setWindowJobsRevision] = useState(0);
  const refreshWindowJobs = useCallback(() => setWindowJobsRevision(value => value + 1), []);
  return <div>
    <ScanCoverageCalendar revision={windowJobsRevision} />
    <DirectoryScans onComplete={refreshWindowJobs} />
    <ModelScans />
    <JobCompletionEstimate revision={windowJobsRevision} />
    <WindowJobs revision={windowJobsRevision} onComplete={refreshWindowJobs} />
  </div>;
}
