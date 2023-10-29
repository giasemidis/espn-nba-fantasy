
# import sys
import streamlit as st

# sys.path.append('.')
from global_params import ROUND_HELP  # noqa: E402
from EspnFantasyMatchUp import EspnFantasyMatchUp  # noqa: E402
from utils.get_logger import get_logger  # noqa: E402
from utils.streamlit_utils import get_cookies_league_params  # noqa: E402

logger = get_logger(__name__)


# @st.cache
def get_round_params():
    round = st.number_input(label="Round", min_value=1,
                            max_value=None, value=1, step=1, help=ROUND_HELP)
    start_date = st.date_input(
        label="Start date",
        help="The start date of the simulation/analysis",
        format="YYYY-MM-DD"
    )
    end_date = st.date_input(
        label="End date",
        help="The end date of the simulation/analysis",
        format="YYYY-MM-DD"
    )
    home_team = st.text_input(
        label="Home team", help="Abbreviation of the home team")
    away_team = st.text_input(
        label="Away team", help="Abbreviation of the away team")
    use_current_score = st.toggle(label="Use current score", help="")
    # valid values: season average, last 7 days average, last 15 days average,
    # last 30 days average, season's projections, previous season average
    stat_types = [
        "season average",
        "last 7 days average",
        "last 15 days average",
        "last 30 days average",
        "season's projections",
        "previous season average"
    ]
    stat_type = st.selectbox(label="Stat type", options=stat_types, help="")
    round_params = {
        "round": round,
        "start_date": start_date,
        "end_date": end_date,
        "home_team": home_team,
        "away_team": away_team,
        "stat_type": stat_type,
        "use_current_score": use_current_score
    }
    return round_params


def main():
    st.title("Head-to-head pre-round analysis")
    app_description = """
    This app helps the user to prepare for an upcoming match-up between two fantasy
    teams. This is based on a league with *Head to Head Each Category* scoring type
    and 9 statistical categories (FG%, FT%, 3PM, REB, AST, STL, BLK, TO, PTS).

    * It compares their schedule (number of starter players and unused players)
    * It compares the teams' historic stats up to this round
    * Simulates/projects the match-up based on the players' average stats and schedule.
    * Allows for scenarios, such as replace player X with player Y

    Further details on [this Medium blog post](https://g-giasemidis.medium.com/nba-fantasy-analytics-with-python-on-epsn-f03f10a60187).

    Use this *public* league id `10149515` for trying out app.
    No need for `swid` and `espn_s2` cookies. This league is based on the same
    nine aforementioned stats, but uses a *Head to Head Points* scoring system. Here, the league is emulated as if the scoring system was "Head to Head Each Category". Checkout the [Post round analysis](https://espn-nba-fantasy.herokuapp.com/) app for the participated teams and their abbreaviations.

    Report bugs and issues [here](https://github.com/giasemidis/espn-nba-fantasy/issues).

    """
    with st.expander("App Description", expanded=True):
        st.markdown(app_description)

    cookies, league_settings = get_cookies_league_params()
    with st.form(key='my_form'):
        round_params = get_round_params()
        submit_button = st.form_submit_button(label='Submit')

    use_current_score = round_params.pop("use_current_score")
    if submit_button:
        with st.spinner('We are doing the clever stuff'):
            espn = EspnFantasyMatchUp(cookies, league_settings, **round_params)
            h2h_stat_df = espn.h2h_season_stats_comparison().astype("O")
            schedule_df = espn.compare_schedules().astype("O")
            sim_df = espn.simulation(use_current_score=use_current_score)

        st.text("Navigate across the tabs to access the different analysis tables")
        tab1, tab2, tab3 = st.tabs(
            ["H2H season stats comparison", "Schedule Comparison", "Simulation"]
        )

        with tab1:
            st.dataframe(h2h_stat_df)

        with tab2:
            st.dataframe(schedule_df)

        with tab3:
            st.dataframe(sim_df)

    return


if __name__ == "__main__":
    main()
