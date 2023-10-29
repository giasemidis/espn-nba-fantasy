
import sys
import streamlit as st

# sys.path.append('.')

from EspnFantasyRoundAnalysis import EspnFantasyRoundAnalysis  # noqa: E402
from global_params import DTYPES, ROUND_HELP, SCORING_PERIOD_HELP
from utils.get_logger import get_logger  # noqa: E402
from utils.streamlit_utils import get_cookies_league_params  # noqa: E402

logger = get_logger(__name__)


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def get_round_params():
    st.header("Round Parameters")
    week = st.number_input(
        "The round for analysis",
        value=0,
        min_value=0,
        max_value=100,
        step=1,
        help=ROUND_HELP
    )

    scoring_period = st.number_input(
        "Scoring period (Optional)",
        value=0,
        min_value=0,
        max_value=500,
        step=1,
        help=SCORING_PERIOD_HELP
    )
    scoring_period = None if scoring_period == 0 else scoring_period
    round_params = {
        "round": week,
        "scoring_period": scoring_period
    }
    return round_params


def main():

    # Start the App #
    st.title("Post round analysis for ESPN NBA Fantasy leagues")

    app_desc_body = """
        This app assesses the teams' performance of a past round. This is based on
        a league with *Head to Head Each Category* scoring type and 9 stat
        categories (FG%, FT%, 3PM, REB, AST, STL, BLK, TO, PTS).

        * It displays the 8 stat categories in addition to total minutes and games
            played by each team in a single table.
        * The ranking of the above stat categories, i.e. how each team is ranked
            for each category.
        * The score differential for all possible match-ups in the round. It gives
            a sense of how a team performed compared to the whole league, not just a
            single match-up.
        * The percentage wins of all possible match-ups. For example, a team might
            have won its match-up, but its percentage win is very low. This
            indicates a weak/lucky team which played against an even weaker
            opponent. Changes are still required. On the other hand, if a team lost
            a match-up, but its percentage win is high, it indicates a strong team
            which happened to play against the strongest opponent. Changes might
            not be required.

        Further details on [this Medium blog post](https://g-giasemidis.medium.com/nba-fantasy-analytics-with-python-on-epsn-f03f10a60187).

        Use this *public* league id `10149515` and year 2022 for trying out app.
        No need for `swid` and `espn_s2` cookies. This league is based on the same
        nine aforementioned stats, but uses a *Head to Head Points* scoring system.
        Here, the league is emulated as if the scoring system was
        "Head to Head Each Category".

        Report bugs and issues
        [here](https://github.com/giasemidis/espn-nba-fantasy/issues).
    """  # noqa: E501

    with st.expander("App Description", expanded=True):
        st.markdown(app_desc_body)

    app_parm_body = """
        Fill in the parameters for the app to run and click on "Submit" button.
    """
    st.text(app_parm_body)

    with st.form(key='my_form'):
        cookies, league_settings = get_cookies_league_params()
        round_params = get_round_params()

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        with st.spinner('We are doing the clever stuff'):
            espn = EspnFantasyRoundAnalysis(
                cookies=cookies,
                league_settings=league_settings,
                **round_params
            )
            adv_stats_df = espn.get_adv_stats_of_round().astype(DTYPES)
            adv_stats_rank_df = espn.compute_stats_ranking_of_round()
            h2h_df = espn.compute_h2h_score_table()
            win_ratio_df = espn.win_ratio_in_round()

        st.text("Navigate across the tabs to access the different analysis tables")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Round Stats", "Stat Ranking", "Head to Head", "% Wins"]
        )

        with tab1:
            st.header("Stats of the round for all fantasy teams")
            st.text(
                """
                9 basic ESPN fantasy stats, in addition to total minutes and games
                for the round under consideration.
                """
            )
            st.table(data=adv_stats_df)
            # st.download_button(
            #     "Download data",
            #     data=convert_df(adv_stats_df),
            #     file_name="round_stats.csv"
            # )

        with tab2:
            st.header("Ranking index for each statistical category")
            st.text(
                """
                Ranking of the teams for the aforementioned statistical categories
                """
            )
            st.table(data=adv_stats_rank_df)
            # st.download_button(
            #     "Download data",
            #     data=convert_df(adv_stats_rank_df),
            #     file_name="round_stats_ranking.csv"
            # )

        with tab3:
            st.header("Head to Head scores for all possible match-ups in the round")
            st.text(
                """
                Score differential for the round under consideration of each team in
                the league against each other team.
                """
            )
            st.table(data=h2h_df)
            # st.download_button(
            #     "Download data",
            #     data=convert_df(h2h_df),
            #     file_name="head_to_head_results.csv"
            # )

        with tab4:
            st.header("Percentage wins from the H2H matchups")
            st.text(
                """
                Based on the previous head to head scores, what is the percentage of
                winsfor and average differential score for each team?
                """
            )
            st.table(data=win_ratio_df)
            # st.download_button(
            #     "Download data",
            #     data=convert_df(win_ratio_df),
            #     file_name="percentage_wins.csv"
            # )

        st.success('Done!')


if __name__ == "__main__":
    main()
