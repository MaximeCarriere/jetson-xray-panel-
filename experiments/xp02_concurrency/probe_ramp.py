"""Find the process-per-model concurrency ceiling on this 8GB board.

Sweeps N upward, stops at the first N that fails (memory wall). Prints available
RAM before each run so we can see the headroom collapse.
"""
import sys

import runner_concurrent as rc


def avail_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) // 1024
    return -1


def main():
    ns = [int(x) for x in sys.argv[1:]] or [5, 6, 7]
    for n in ns:
        print(f"--- N={n}  (avail {avail_mb()} MB before) ---", flush=True)
        try:
            r = rc.run_concurrent(["densenet121-res224-all"] * n,
                                  duration=4.0, warmup=10)
            thr = r["n_images"] / r["wall_s"]
            pw = [w["throughput_ips"] for w in r["per_worker"]]
            print(f"N={n}: {thr:.1f} img/s | per-worker {pw} | "
                  f"avail {avail_mb()} MB after", flush=True)
        except Exception as e:
            print(f"N={n}: FAILED ({type(e).__name__}: {e}) | "
                  f"avail {avail_mb()} MB", flush=True)
            break


if __name__ == "__main__":
    main()
