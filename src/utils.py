def prompt_validation_or_new_threshold(message=None):
    """
    Prompt user for "yes" or "exit" or a float value between 0 and 1.

    :param message: [str] Message to be printed to the user.
    :return:        [None] If user input is none of the above, return None.
                    [str] If user input is "yes" or "exit", return "yes" or "exit".
                    [float] If user input is a valid float between 0 and 1, return the value as a float.
    """
    if message is None:
        message = "Type 'yes' to confirm OR 'exit' OR enter a new threshold value (val between 0 and 1).\n"
    end = input(message)
    if end == "exit" or end == "yes":
        return end
    try:
        threshold = float(end)
        if not (0 <= threshold <= 1):
            raise ValueError(f"{threshold} is not a valid threshold. Threshold shall be between 0 and 1")
    except ValueError:
        print(f"{end} is not a valid threshold. Threshold shall be between 0 and 1")
        return None
    return threshold


