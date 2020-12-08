import pandas as pd  
import matplotlib.pyplot as plt


def read_stats(file_name):
    df = pd.read_csv(file_name, header= None, names=['score'])
    return df


errors = read_stats('qlearning.txt')
ax = errors.plot.line(figsize=(20, 10), ylim=(0, 0.5))
ax.set_xlabel("Simulation Iteartion")
ax.set_ylabel("Q-value Difference")
plt.show()


mpl.rcParams['agg.path.chunksize'] = 10000
corrolated_q_errors = read_stats('corrolatedq.txt')
ax = corrolated_q_errors.plot.line(figsize=(20, 10), ylim=(0, 0.5))
ax.set_xlabel("Simulation Iteartion")
ax.set_ylabel("Q-value Difference")
plt.show()



friend_q_errors = read_stats('friendq.txt')
friend_q_errors = friend_q_errors[friend_q_errors['score'] > 0]
ax = friend_q_errors.plot.line(figsize=(20,10), ylim=(0, 0.5))
ax.set_xlabel("Simulation Iteartion")
ax.set_ylabel("Q-value Difference")
plt.show()



foe_q = read_stats('foeq.txt')
ax = foe_q.plot.line(figsize=(20, 10), ylim=(0, 0.5))
ax.set_xlabel("Simulation Iteartion")
ax.set_ylabel("Q-value Difference")
plt.show()

