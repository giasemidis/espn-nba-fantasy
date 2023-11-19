import streamlit as st

from global_params import ROUND_HELP, SCENARIO_HELP
from EspnFantasyLeague import EspnFantasyLeague
from EspnFantasyMatchUp import EspnFantasyMatchUp
from utils.get_logger import get_logger
from utils.app_utils import get_cookies_league_params, parameter_checks, format_sub_dct

logger = get_logger(__name__)


# @st.cache
def get_round_params(espnfantasyleague_obj):
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
    display_teams_dict = {
        f"{espnfantasyleague_obj.team_id_name_dict[k]} ({v})": v
        for k, v in espnfantasyleague_obj.team_id_abbr_dict.items()
    }
    teams = display_teams_dict.keys()
    home_team = st.selectbox(label="Home team", options=teams, index=None)
    away_team = st.selectbox(label="Away team", options=teams, index=None)
    if home_team is not None and away_team is not None:
        home_team = display_teams_dict[home_team]
        away_team = display_teams_dict[away_team]
    use_current_score = st.toggle(
        label="Use current score", value=True,
        help="Use the actual score up to this moment for an on-going match-up"
    )
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
    stat_type = st.selectbox(
        label="Stat type", options=stat_types,
        help="The type of stat to be used in the simulation of the match-up"
    )
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
    st.title("Head-to-head pre-round analysis for ESPN NBA Fantasy leagues")
    app_description = """
    This app helps the user to prepare for an upcoming match-up between two
    fantasy teams. This is based on a league with *Head to Head Each Category*
    scoring type and 9 statistical categories (FG%, FT%, 3PM, REB, AST, STL,
    BLK, TO, PTS).

    * It compares their schedule (number of starter players and unused players)
    * It compares the teams' historic stats up to this round
    * Simulates/projects the match-up based on the players' average stats and
        schedule.
    * Allows for scenarios, such as replace player X with player Y

    Further details on
    [this Medium blog post](https://g-giasemidis.medium.com/nba-fantasy-analytics-with-python-on-epsn-f03f10a60187).

    Use this *public* league id `10149515` for trying out app.
    No need for `swid` and `espn_s2` cookies. This league is based on the same
    nine aforementioned stats, but uses a *Head to Head Points* scoring system.
    Here, the league is emulated as if the scoring system was
    "Head to Head Each Category".

    Report bugs and issues
    [here](https://github.com/giasemidis/espn-nba-fantasy/issues).
    """
    with st.expander("App Description", expanded=True):
        st.markdown(app_description)

    cookies, league_settings = get_cookies_league_params()

    if league_settings["league_id"] != "":
        league_params_ok = parameter_checks(
            swid=cookies["swid"],
            espn_s2=cookies["espn_s2"],
            league_id=league_settings["league_id"]
        )
        if league_params_ok:
            espn = EspnFantasyLeague(cookies, league_settings)

        with st.form(key='my_form'):
            round_params = get_round_params(espn)
            with st.expander("Scenario subs", expanded=False):
                st.write(
                    """Explore hypothetical player substitutions;
                    adding and/or removing players and their impact
                    on schedule and the simulated statistics"""
                )
                home_player_add = st.text_area(
                    "Home players to add", value="", help=SCENARIO_HELP)
                home_player_rmv = st.text_area(
                    "Home players to remove", value="", help=SCENARIO_HELP)
                away_player_add = st.text_area(
                    "Away players to add", value="", help=SCENARIO_HELP)
                away_player_rmv = st.text_area(
                    "Away players to remove", value="", help=SCENARIO_HELP)
                home_scn_players = {
                    "add": format_sub_dct(home_player_add),
                    "remove": format_sub_dct(home_player_rmv),
                }
                away_scn_players = {
                    "add": format_sub_dct(away_player_add),
                    "remove": format_sub_dct(away_player_rmv),
                }
            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            if (round_params["home_team"] is not None) \
                    and (round_params["away_team"] is not None):
                if round_params["home_team"] == round_params["away_team"]:
                    st.warning("Select different teams", icon="⚠️")
                else:
                    with st.spinner('We are doing the clever stuff'):
                        use_current_score = round_params.pop(
                            "use_current_score")
                        espn = EspnFantasyMatchUp(
                            cookies, league_settings, **round_params,
                            home_scn_players=home_scn_players,
                            away_scn_players=away_scn_players
                        )
                        h2h_stat_df = (
                            espn.h2h_season_stats_comparison().astype("O")
                        )
                        schedule_df = espn.compare_schedules().astype("O")
                        sim_df = espn.simulation(
                            use_current_score=use_current_score)

                    st.text(
                        "Navigate across the tabs to access the different "
                        "analysis tables"
                    )
                    tab1, tab2, tab3 = st.tabs(
                        ["H2H season stats comparison",
                            "Schedule Comparison", "Simulation"]
                    )

                    with tab1:
                        st.dataframe(h2h_stat_df)

                    with tab2:
                        st.dataframe(schedule_df)

                    with tab3:
                        st.dataframe(sim_df)
            else:
                st.warning("Select home and away teams", icon="⚠️")

    return


if __name__ == "__main__":
    main()
