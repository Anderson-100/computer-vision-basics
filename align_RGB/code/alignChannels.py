import numpy as np

def alignChannels(img, max_shift):
    # pred_shift is a 2x2 array with
    # pred_shift[0,0] -- shift needed along i (y axis) for G to align with R
    # pred_shift[0,1] -- shift needed along j (x axis) for G to align with R
    # pred_shift[1,0] -- shift needed along i (y axis) for B to align with R
    # pred_shift[1,1] -- shift needed along j (x axis) for B to align with R
    crop = 15
    red_ch = img[:, :, 2]

    pred_shift = np.zeros((2,2))

    # Compare blue first, then green one at a time
    for ch in range(2):
        compare_ch = img[:, :, ch]
        red_cropped = red_ch[crop:-crop, crop:-crop]
        max_sim = float('-inf')

        # Compare within max_shift range
        for i in range(-max_shift[0], max_shift[1] + 1):
            for j in range(-max_shift[0], max_shift[1] + 1):
                # Shift and crop the current color we're comparing
                compare_ch_shifted = np.roll(compare_ch, i, axis=0)
                compare_ch_shifted = np.roll(compare_ch_shifted, j, axis=1)
                compare_cropped = compare_ch_shifted[crop:-crop, crop:-crop]

                sim = ch_similarity(red_cropped, compare_cropped)
            
                # Check if this is better than the best we've done
                if sim > max_sim:
                    max_sim = sim
                    pred_shift[1-ch][0] = i
                    pred_shift[1-ch][1] = j

    # # Apply the estimated shift to the G and B channels to align

    aligned_image = img.copy()

    # blue first, then green
    for ch_num in range(len(pred_shift)):
        for axis in range(len(pred_shift[0])):
            aligned_image[:,:,ch_num] = np.roll(aligned_image[:,:,ch_num], int(pred_shift[1-ch_num][axis]), axis=axis)

    return(aligned_image, pred_shift)

def ch_similarity(ch1, ch2):
    ch1_flat = ch1.flatten()
    ch2_flat = ch2.flatten()

    cos_theta = np.dot(ch1_flat.T, ch2_flat) / (np.linalg.norm(ch1_flat) * np.linalg.norm(ch2_flat))

    return cos_theta