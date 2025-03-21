# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.

## POLICY ITERATION ALGORITHM
-> Step1 :
we are going to do policy evaluation of each state to get the state value function where the initial policy is defined randomly to the mdp.

-> Step2:
Once we obtain convergence in the policy evaluation then implement policy improvement where we are going to find best optimal policy until the previous and current policy are same.
</br>
</br>

## POLICY IMPROVEMENT FUNCTION
#### Name : KULASEKARAPANDIAN K
#### Register Number : 212222240052
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to improve the given policy
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
          new_pi=lambda s:{s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi
```
## POLICY ITERATION FUNCTION
#### Name : KULASEKARAPANDIAN K
#### Register Number : 212222240052
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
   random_actions=np.random.choice(tuple(P[0].keys()),len(P))
   pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
   while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
   return V, pi
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
![Screenshot 2025-03-19 115151](https://github.com/user-attachments/assets/e5f72271-df6e-4ce2-b1a6-7d9b930d6e78)
![Screenshot 2025-03-19 115124](https://github.com/user-attachments/assets/5ca0a2ce-dae1-4074-8b8c-514736c940c1)
![Screenshot 2025-03-19 115139](https://github.com/user-attachments/assets/86555093-77c8-4208-b7b3-71ced15cd7eb)
</br>
</br>

### 2. Policy, Value function and success rate for the Improved Policy
![Screenshot 2025-03-19 115231](https://github.com/user-attachments/assets/9ffd5ccf-68f3-45d0-bac4-02c105689373)
![Screenshot 2025-03-19 115205](https://github.com/user-attachments/assets/a4e4c24d-0aa4-4750-a8db-4ca0c0d29468)
![Screenshot 2025-03-19 115217](https://github.com/user-attachments/assets/aef22232-f157-4f4f-8802-95a547bd1a87)
</br>
</br>

### 3. Policy, Value function and success rate after policy iteration
![Screenshot 2025-03-19 115309](https://github.com/user-attachments/assets/7cacee97-bc0a-4839-9807-c88b165441d1)
![Screenshot 2025-03-19 115247](https://github.com/user-attachments/assets/b3a0ad74-c26f-45d1-ba6b-db1349458cea)
![Screenshot 2025-03-19 115300](https://github.com/user-attachments/assets/30c657d9-20c9-4ddc-b974-f233c4e3097b)
</br>
</br>

## RESULT:
Thus, The Python program to find the optimal policy for the given MDP using the policy iteration algorithm is successfully executed.
