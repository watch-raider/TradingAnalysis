import numpy as np
import math

def calculate_accuracy(predictions, actual_values, day_in_advance):
    p = predictions[:-day_in_advance]
    av = actual_values[:-day_in_advance]
    
    rsme = calculate_rsme(p, av)
    pr = calculate_pearsons_r(p, av)
    
    return rsme, pr


def calculate_pearsons_r(p_aligned, av_aligned):
    # p_diffs = p_aligned - np.mean(p_aligned)
    # av_diffs = av_aligned - np.mean(av_aligned)
    # multiplied_diffs = p_diffs * av_diffs
    # p_diffs_squared = p_diffs**2
    # av_diffs_squared = av_diffs**2
    # p_diffs_squared

    # pr = sum(multiplied_diffs) / (math.sqrt(sum(p_diffs_squared)) * math.sqrt(sum(av_diffs_squared)))

    # return round(pr, 2) 
    return round(np.corrcoef(p_aligned, av_aligned)[0][1], 2)

def calculate_rsme(p_aligned, av_aligned) :
    diffs = p_aligned - av_aligned
    squared_diffs = diffs**2
    squared_error = sum(squared_diffs)
    mean_squared_error = squared_error / len(p_aligned)
    root_mean_squared_error = math.sqrt(mean_squared_error)

    return round(root_mean_squared_error, 2)