import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from utils.utils import fantasy_matchup_score
from EspnFantasyLeague import EspnFantasyLeague


class EspnFantasyRoundAnalysis(EspnFantasyLeague):
    def __init__(self, cookies, league_settings,
                 round=None, scoring_period=None):
        super().__init__(cookies, league_settings)

        if round is None:
            self.round = \
                self.division_setting_data['status']['currentMatchupPeriod'] - 1
        else:
            self.round = round
        self.scoring_period = scoring_period

        self.fantasy_teams_data = None
        self.stats_all_rounds = None
        self.adv_stats_of_round = None
        self.stats_ranking_of_round = None
        self.h2h_score_table = None
        self.aggr_round_scores = None

    def get_adv_stats_of_round(self):
        '''
        Filters stats by round (`week`)
        '''
        if self.stats_all_rounds is None:
            table = self.get_fantasy_team_stats_per_round()
        else:
            table = self.stats_all_rounds

        mints_games_round_df = self.get_adv_stats_per_fantasy_team(
            self.round, self.scoring_period
        )

        table = table.loc[table['Round'] == self.round, self.fantasy_stats]

        table = table.merge(
            mints_games_round_df, left_index=True, right_index=True
        )
        self.adv_stats_of_round = table
        return table

    def compute_stats_ranking_of_round(self):
        '''
        Ranking index for each statistical category.
        '''
        if self.adv_stats_of_round is None:
            stat_table = self.get_adv_stats_of_round()
        else:
            stat_table = self.adv_stats_of_round

        ranks_df = pd.DataFrame([], index=stat_table.index)
        for stat in stat_table.columns:
            if stat == 'TO':
                ranks_df[stat] = stat_table[stat].argsort().argsort().values + 1
            else:
                ranks_df[stat] = (
                    stat_table[stat].argsort()[::-1].argsort().values + 1
                )
        self.stats_ranking_of_round = ranks_df
        return ranks_df

    def compute_h2h_score_table(self):
        '''
        Computes the score matrix, i.e. the match-up score between
        all teams against all other teams, for a given week
        '''
        if self.adv_stats_of_round is None:
            stat_table = self.get_adv_stats_of_round()
        else:
            stat_table = self.adv_stats_of_round.copy()

        stat_table = stat_table[self.fantasy_stats]
        # get the team abbreviations for the filtered dataframe.
        teams = stat_table.index
        # get the stat values of
        X = stat_table.values
        # find the index of the 'TO' (to correct)
        index_to = np.where(stat_table.columns == 'TO')[0][0]

        # temporary metric function using the `fantasy_matchup_score`
        # and the `index_to`
        metric = lambda x, y: fantasy_matchup_score(x, y, index_to)
        # make the square scores matrix
        dist_mat = squareform(pdist(X, metric=metric))
        # the metric function is symmetric, correct it and make it
        # anti-symmetric (if team A wins team B, e.g. +2, team B loses from
        # team A, i.e. -2)
        dist_mat[np.tril_indices(dist_mat.shape[0])] = (
            - dist_mat[np.tril_indices(dist_mat.shape[0])])
        # make the scores matrix into a dataframe
        df = pd.DataFrame(dist_mat, index=teams, columns=teams).astype(int)

        self.h2h_score_table = df
        return df

    def win_ratio_in_round(self):
        '''
        Computes aggregates scores (% of Wins) for each fantasy team from
        their performance against every other team (h2h score table)
        '''
        if self.h2h_score_table is None:
            score_mat = self.compute_h2h_score_table()
        else:
            score_mat = self.h2h_score_table.copy()

        # get the number of wins against all other teams
        wins = (score_mat > 0).sum(axis=1)
        # get the percentage of wins
        perc = wins / (score_mat.shape[0] - 1)
        # get the average score against all other teams
        avg = score_mat.sum(axis=1) / (score_mat.shape[0] - 1)
        # make a dataframe
        scores_df = pd.DataFrame({
            'Wins': wins, 'Wins%': perc, 'Avg Score': avg
        })

        aggr_tbl = (
            scores_df.rename(index=self.team_abbr_name_dict)
            .sort_values(['Wins', "Avg Score"], ascending=False)
        )
        self.aggr_round_scores = aggr_tbl
        return aggr_tbl
