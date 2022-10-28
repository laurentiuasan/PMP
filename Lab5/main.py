from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx


if __name__ == '__main__':
    game_model = BayesianNetwork(
        [
            ("Card1", "Decision1"),
            ("Decision1", "Decision2"),
            ("Card2", "Decision2"),
            ("Decision2", "Decision3"),
            ("Card1", "Decision3"),
            ("Card1", "Result"),
            ("Card2", "Result"),
            ("Decision1", "Result"),
            ("Decision2", "Result"),
            ("Decision3", "Result")
        ]
    )

    cpd_card1 = TabularCPD(
        variable='Card1', variable_card=5, values=[[1/5], [1/5], [1/5], [1/5], [1/5]]   # AC, KC, KH, QC, QH
    )

    cpd_card2 = TabularCPD(
        variable='Card2',
        variable_card=5,
        values=[[0, 1/20, 1/20, 1/20, 1/20],    # KC, KH, QC, QH
                [1/20, 0, 1/20, 1/20, 1/20],    # AC, KH, QC, QH
                [1/20, 1/20, 0, 1/20, 1/20],    # AC, KQ, QC, QH
                [1/20, 1/20, 1/20, 0, 1/20],    # AC, KQ, KH, QH
                [1/20, 1/20, 1/20, 1/20, 0]],   # AC, KQ, Kh, QC
        evidence=["Card1"],
        evidence_card=[5]
    )

    cpd_decision1 = TabularCPD(
        variable="Decision1",
        variable_card=2,    # check = 0, bet = 1
        values=[[0, 1], [0.1, 0.9], [0.2, 0.8], [0.5, 0.5], [1, 0]],
        evidence=["Card1"],
        evidence_card=5
    )

    cpd_decision2 = TabularCPD(
        variable="Decision2",
        variable_card=2,    # check/fold = 0, bet = 1
        values=[[0, 1/4, 2/4, 3/4, 1, 1, 3/4, 2/4, 1/4, 0],     # D2 = 0, D1 = {0,1}, C2={AC,KC,KH, QC, QH}-C1
                [1, 3/4, 2/4, 2/4, 0, 1, 3/4, 2/4, 1/4, 0]],    # D2 = 1, D1 = {0,1}, C2 = {AC, KC, KH, QC, QH} - C1
        evidence=["Decision1", "Card2"],
        evidence_card=[2, 5]
    )

    cpd_decision3 = TabularCPD(
        variable="Decision3",
        variable_card=2,
        values=[[0, 0, 0, 0, 0, 0, 1/4, 2/4, 3/4, 1],     # D3 = 0, D2 = {0,1}, C1 = {AC,KC,KH, QC, QH}
                [0, 0, 0, 0, 0, 1, 3/4, 2/4, 1/4, 0]],    # D3 = 1, D2 = {0,1}, C1 = {AC,KC,KH, QC, QH}
        evidence=["Decision2", "Card1"],
        evidence_card=[2, 5]
    )

    cpd_result = TabularCPD(
        variable="Result",
        variable_card=2,
        values=[],
        evidence=["Card1", "Card2", "Decision1", "Decision2", "Decision3"],
        evidence_card=[5, 5, 2, 2, 2]
    )