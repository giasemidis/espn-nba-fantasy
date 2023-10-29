import sys
import streamlit as st
from global_params import (
    CURRENT_SEASON, SWID_HELP, ESPN_S2_HELP, LEAGUE_ID_HELP, SEASON_HELP, ROUND_HELP
)

sys.path.append('.')
from src.EspnFantasyMatchUp import EspnFantasyMatchUp  # noqa: E402


# @st.cache
def get_cookies_league_params():
    with st.sidebar:
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
    return cookies, league_params


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
    cookies, league_settings = get_cookies_league_params()
    with st.form(key='my_form'):
        round_params = get_round_params()
        submit_button = st.form_submit_button(label='Submit')

    use_current_score = round_params.pop("use_current_score")
    if submit_button:
        # st.write(cookies)
        # st.write(league_settings)
        # st.write(type(round_params["end_date"]))
        espn = EspnFantasyMatchUp(
            cookies, league_settings,
            **round_params
        )

        st.text("Navigate across the tabs to access the different analysis tables")
        tab1, tab2, tab3 = st.tabs(
            ["H2H season stats comparison", "Schedule Comparison", "Simulation"]
        )
        h2h_stat_df = espn.h2h_season_stats_comparison().astype("O")
        schedule_df = espn.compare_schedules().astype("O")

        with tab1:
            st.dataframe(h2h_stat_df)

        with tab2:
            st.dataframe(schedule_df)

        with tab3:
            st.write("This is the simulation tab")
            # df3 = espn.simulation(use_current_score=use_current_score)

    return


if __name__ == "__main__":
    main()
