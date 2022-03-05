import re
import json


def convert_input_strs_to_scn_dict(add, remove):
    """
    Convert the `add` and `remove` strings into
    the appropiate dictionary for scenarioss
    """
    out_dict = {
        "add": convert_input_str_to_dict(add),
        "remove": convert_input_str_to_dict(remove)
    }
    return out_dict


def reformat_str_with_python_symbols(input_str):
    """
    Input parameter field does not allow for python symbols suchs as square
    brackets or colon. Here, we fill in the input str with such python symbols
    before give them dictionary-like structure. E.g.:
    "Name A, Name B 2022-03-02, 2022-03-01" into:
    Name A: [], Name B: [2022-03-02, 2022-03-01]
    """
    players = re.findall(r"[a-zA-Z \-.]+", input_str)
    players_new = []
    for player in players:
        player_strip = player.strip(" ").strip("-")
        if len(player_strip) > 0:
            players_new.append(player_strip)

    dates_new = []
    for i, player in enumerate(players_new):
        start = re.search(player, input_str).span()[0]
        if i < len(players_new) - 1:
            end = re.search(players_new[i + 1], input_str).span()[0]
        else:
            end = len(input_str)
        substring = input_str[start:end]
        date_strip = substring.replace(player, "")
        date_strip = date_strip.strip(" ").strip(",")
        if date_strip == "":
            date_strip = "[]"
        else:
            date_strip = date_strip.replace(date_strip, '[%s]' % date_strip)
        dates_new.append(date_strip)

    if len(players_new) != len(dates_new):
        print("ERROR", players_new, dates_new)

    out_str = "".join(
        "{0}: {1}, ".format(p, d)
        for p, d in zip(players_new, dates_new)
    )
    out_str = out_str.strip(", ")
    return out_str


def convert_input_str_to_dict(input_str):
    '''
    Convert an input string of player names and dates
    into a dictionary
    '''

    if input_str != "":
        if re.search(r'[:\[\]]', input_str) is None:
            input_str = reformat_str_with_python_symbols(input_str)
        # First identify the players and enclose their names in
        # double quotation marks
        # Allow for dots and dash in player names
        players = re.findall(r"[a-zA-Z \-.]+", input_str)
        for player in players:
            player = player.strip(" ").strip("-")
            if len(player) > 0:
                input_str = input_str.replace(player, '"%s"' % player)

        # Identify the dates and enclose them in double quotation marks.
        dates = re.findall(r"20[0-9][0-9]-[0-1][0-9]-[0-3][0-9]", input_str)
        dates = set(dates)
        for date in dates:
            input_str = input_str.replace(date, '"%s"' % date)

    # Finally enclose the whole string into curly brackets
    # for reading it as a dictionary with json
    input_str = "{%s}" % input_str
    return json.loads(input_str)
