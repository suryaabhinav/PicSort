export type StartRunRequest = {
    root: string;
    batch_size?: number;
    run_faces?: boolean;
    // add more config knobs later…
};

export type StartRunResponse = { run_id: string };

export type RunStatus = {
    run_id: string;
    stage: string;
    progress: number;
    message: string;
    error?: string;
    timings?: Record<string, number>;
    paths?: Record<string, string>;
    root?: string;
};
