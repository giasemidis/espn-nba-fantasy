import sys
import streamlit as st
from global_params import (
    CURRENT_SEASON, SWID_HELP, ESPN_S2_HELP, LEAGUE_ID_HELP, SEASON_HELP, ROUND_HELP
)

# sys.path.append('.')
from utils.app_utils import parameter_checks  # noqa: E402
from utils.get_logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def get_cookies_league_params():
    with st.sidebar:
        st.header("Cookie and league parameters")
        st.write(
            "Check the help button for further details and how to identify the "
            "cookies and the league id."
        )
        swid = st.text_input(label="swid (cookie)", help=SWID_HELP)
        espn_s2 = st.text_input(label="espn_s2 (cookie)", help=ESPN_S2_HELP)
        league_id = st.text_input(
            label="The ID of the ESPN league", help=LEAGUE_ID_HELP)
        season = st.number_input(
            'The season of the league',
            value=CURRENT_SEASON,
            min_value=2019,
            max_value=CURRENT_SEASON,
            step=1,
            help=SEASON_HELP
        )
    cookies = {
        "swid": swid,
        "espn_s2": espn_s2,
    }
    league_params = {
        "league_id": league_id,
        "season": season
    }
    parameter_checks(swid, espn_s2, league_id)
    return cookies, league_params
