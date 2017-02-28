import numpy as np
import ipdb
"""
mdp.P is a two-level dict where the first key is the state and the second key
is the action.
mdp.P[state][action] is a list of tuples (probability, nextstate, reward)
For example, state 0 is the initial state, and the transition information for
s=0, a=0 is P[0][0] = mdp.P[0][0].

As another example, state 5 corresponds to a hole in the ice, which
transitions to itself with probability 1 and reward 0.

"""

class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means (used for plotting)


def get_sample_mdp(alpha = 0.99):
    P = {0: {0: [(alpha, 0, 0.0), (1.0 - alpha, 1, 0.0)],
             1: [(1.0, 0, 1.0)]},
         1: {0: [(alpha, 1, 0.0), (1.0 - alpha, 2, 0.0)],
             1: [(1.0, 1, 1.0)]},
         2: {0: [(1.0, 2, 0.0)],
             1: [(1.0, 2, 6.5)]}}
    n_states = 3
    n_actions = 2
    desc = "MDP that converges real slow"
    mdp = MDP(P, n_states, n_actions, desc)
    return mdp


def value_iteration(mdp, gamma, nIt):
    print("Iteration | max|V-Vprev| | # chg actions | V[0]")
    print("----------+--------------+---------------+---------")
    Vs = [np.zeros(mdp.nS)] 
    pis = []
    for it in range(nIt):
        Vprev = Vs[-1]
        oldpi = pis[-1] if len(pis) > 0 else None
        # Your code should define variables V: the bellman backup applied to Vprev
        # and pi: the greedy policy applied to Vprev
        pi = np.zeros(mdp.nS)
        V = np.zeros(mdp.nS)
        
        for state in range(mdp.nS):
            per_action_value = np.zeros(mdp.nA)
            for action in range(mdp.nA):
                acc = 0
                possible_transitions = mdp.P[state][action]
                for prob, next_state, reward in possible_transitions:
                    acc += prob * (reward + gamma * Vprev[next_state])
                per_action_value[action] = acc
            pi[state] = np.argmax(per_action_value)
            V[state] = np.max(per_action_value)
        Vs.append(V)
        pis.append(pi)
        
        
        max_diff = np.abs(V - Vprev).max()
        nChgActions="N/A" if oldpi is None else (pi != oldpi).sum()
        print("%4i      | %6.5f      | %4s          | %5.3f"%(it, max_diff, nChgActions, V[0]))
        Vs.append(V)
        pis.append(pi)
    return Vs, pis


def run_value_iteration_experiment():
    GAMMA = 0.95 
    mdp = get_sample_mdp()
    Vs_VI, pis_VI = value_iteration(mdp, gamma=GAMMA, nIt=65)


def compute_vpi(pi, mdp, gamma):
    r = np.zeros(mdp.nS, dtype=np.float32)
    P = np.zeros((mdp.nS, mdp.nS), dtype=np.float32)

    for state in range(mdp.nS):
        action = pi[state]
        possible_transitions = mdp.P[state][action]
        for prob, next_state, reward in possible_transitions:
            P[state][next_state] = prob
            r[state] += prob * reward
    vpi = np.linalg.inv(np.eye(len(P)) - gamma * P).dot(r)
    return vpi


if __name__ == "__main__":

    mdp = get_sample_mdp()
    gamma = 0.95 
    pi = [1, 1, 1]

    print compute_vpi(pi, mdp, gamma)
