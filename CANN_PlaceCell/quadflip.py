import numpy as np

def quadflip(the_array):
    """
    Quadrant reversing trick for plotting inhibition and excitation kernels.
    Assumes square and even matrix.
    """

    half_row = len(the_array) // 2
    half_col = len(the_array[0]) // 2

    # Extract quadrants
    upper_left = the_array[:half_row, :half_col]
    upper_right = the_array[:half_row, half_col:]
    lower_right = the_array[half_row:, half_col:]
    lower_left = the_array[half_row:, :half_col]

    # Flip each quadrant
    upper_left = np.flipud(np.fliplr(upper_left))
    upper_right = np.flipud(np.fliplr(upper_right))
    lower_right = np.flipud(np.fliplr(lower_right))
    lower_left = np.flipud(np.fliplr(lower_left))

    # Concatenate quadrants
    top = np.concatenate([upper_left, upper_right], axis=1)
    bottom = np.concatenate([lower_left, lower_right], axis=1)
    new_array = np.concatenate([top, bottom], axis=0)

    return new_array


if __name__ == "__main__":
    test_array = np.array([
        [1, 10,421, 21, 27],
        [12, 14,42, 25, 27],
        [33, 31,321, 47, 48],
        [356, 34,5, 43, 42]
    ])

    # Apply the quadflip function
    flipped_array = quadflip(test_array)

    # Display the original and flipped arrays
    print("Original Array:\n", test_array)
    print("\nFlipped Array:\n", flipped_array)
