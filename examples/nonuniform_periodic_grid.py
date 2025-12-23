import numpy as np

def nonuniform_periodic_grid(periods=(1.0, 1.0),
                             a0=1e-9,       # initial tiny step
                             amax=1e-2,     # coarse step
                             refine_window=0.05):  # ramp duration
    times = [0.0]
    t = 0.0

    # growth factor for geometric ramp
    r = (refine_window - a0) / (refine_window - amax)
    if r <= 1.0:
        r = 1.1

    period_start = 0.0

    for p in periods:
        # reset t to the start of the period
        t = period_start

        # 1) first 5 steps use a0
        for _ in range(5):
            step = a0
            # stop if stepping out of the refine window
            if t + step > period_start + refine_window:
                break
            t += step
            times.append(t)

        # 2) geometric ramp until refine_window
        k = 0
        while True:
            step = a0 * (r ** k)
            if t + step >= period_start + refine_window:
                break
            t += step
            times.append(t)
            k += 1

        # 3) correction to exactly reach refine_window
        corr = (period_start + refine_window) - t
        if corr > 0:
            t += corr
            times.append(t)

        # 4) constant coarse steps
        remaining = (period_start + p) - t
        if remaining > 0:
            n_coarse = int(np.floor(remaining / amax))
            for i in range(n_coarse):
                t += amax
                times.append(t)

            leftover = (period_start + p) - t
            if leftover > 1e-12:
                t += leftover
                times.append(t)

        # move to the next period
        period_start += p

    # final time correction
    total_time = sum(periods)
    if times[-1] < total_time:
        times.append(total_time)
    else:
        times[-1] = total_time

    return np.array(times)


# example run
#t = nonuniform_periodic_grid(periods=(1.0, 1.0))

