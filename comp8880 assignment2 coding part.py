#!/usr/bin/env python
# coding: utf-8

# In[54]:


# codes for Question 1
import numpy as np
from scipy.stats import lognorm
from scipy.optimize import minimize
from scipy.integrate import quad


mu = np.log(2)

# Define the target function to achieve the 80-20 rule
def OPTIMI(sigma):
    # Create a log-normal distribution with the given sigma and scale=exp(mu)
    dist = lognorm(s=sigma, scale=np.exp(mu))
    
    # Calculate the total expected wealth
    total_wealth, _ = quad(lambda x: x * dist.pdf(x), 0, np.inf)
    
    # Find x_0.2, the value below which 20% of observations fall
    x_0_2 = dist.ppf(0.2)
    
    # Calculate the wealth held by the bottom 20%
    wealth_bottom_20, _ = quad(lambda x: x * dist.pdf(x), 0, x_0_2)
    
    # Objective: The difference between actual proportion of wealth held by bottom 20% and target (0.2)
    return abs(wealth_bottom_20 / total_wealth - 0.2)

# Initial guess for sigma
initial_sigma = [0.5]

# Perform the optimization to find the sigma that minimizes the difference to the 80-20 rule
result = minimize(target_function, initial_sigma, method='Nelder-Mead')

if result.success:
    optimal_sigma = result.x[0]
    print(f"Optimal σ for the 80-20 rule: {optimal_sigma:.4f}")
else:
    print("Optimization was unsuccessful. Please check the method or initial guess.")


# In[43]:


# question for 2.2

import numpy as np

# adjacency matrix A
A_1 = np.array([[0, 1, 1, 1, 0, 0],   [0, 0, 0, 1, 1, 0],  [0, 0, 0, 1, 0, 1],  [0, 0, 0, 0, 0, 1],  [0, 0, 0, 1, 0, 1],[1, 1, 0, 0, 0, 0]])
A_2 = np.array([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
A_3 = np.array([[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1],  [0, 0, 0, 0, 0, 1],  [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]])

# Transfer matrix
P_1 = np.zeros_like(A_1, dtype=float)

P_2 = np.zeros_like(A_2, dtype=float)
P_3 = np.zeros_like(A_3, dtype=float)
# damping factor, now is 1 - 0.1 = 0.9
d = 0.9
# total pages
n = 6
#making transfer matrix
for i in range(n):
    outlinks = A_1[i, :].sum()
    if outlinks:
        P_1[i, :] = A_1[i, :] / outlinks
    else:
        P_1[i, :] = 1.0 / n
print(P_1)
for i in range(n):
    outlinks = A_2[i, :].sum()
    if outlinks:
        P_2[i, :] = A_2[i, :] /outlinks
    else:
        P_2[i, :] = 1.0 /n
for i in range(n):
    outlinks = A_3[i, :].sum()
    if outlinks:
        P_3[i, :] = A_3[i, :] / outlinks
    else:
        P_3[i, :] = 1.0 / n
P_1 = d * P_1 + (1 - d) / n * np.ones((n, n))
print(P_1)
P_2 = d * P_2 + (1 - d) / n * np.ones((n, n))
P_3 = d * P_3 + (1 - d) / n * np.ones((n, n))

# initial values are all same
page_rank_1 = np.ones(n) / n
page_rank_2 = np.ones(n) / n
page_rank_3 = np.ones(n) / n


# interations
for _ in range(1000):
    page_rank_1_UPDATE = P_1.T @ page_rank_1
    if np.allclose(page_rank_1_UPDATE, page_rank_1, atol=1e-6):
        break
    page_rank_1 = page_rank_1_UPDATE
for _ in range(1000):
    page_rank_2_UPDATE = P_2.T @ page_rank_2
    if np.allclose(page_rank_2_UPDATE, page_rank_2, atol=1e-6):
        break
    page_rank_2 = page_rank_2_UPDATE
for _ in range(1000):
    page_rank_3_UPDATE = P_3.T @ page_rank_3
    if np.allclose(page_rank_3_UPDATE, page_rank_3, atol=1e-6):
        break
    page_rank_3 = page_rank_3_UPDATE

print("PageRank_1:", page_rank_1)
print("PageRank_2:", page_rank_2)
print("PageRank_3:", page_rank_3)


# In[24]:


def hits_algorithm(A, convergency=1e-8):
    n = A.shape[0]
    hubs = np.ones(n)
    authorities = np.ones(n)
    
    for _ in range(100):
        update_authorities = A.T @ hubs
        update_hubs = A@ update_authorities
        
        # normalize
        update_authorities /= np.linalg.norm(update_authorities, 2)
        update_hubs /= np.linalg.norm(update_hubs, 2)
        
        # convergent to a small value
        if np.allclose(hubs,update_hubs,atol=convergency) and np.allclose(authorities,update_authorities,atol=convergency):
            break
        
        hubs, authorities = update_hubs, update_authorities
    
    return hubs, authorities
hubs_1, authorities_1 = hits_algorithm(A_1)
hubs_2, authorities_2 = hits_algorithm(A_2)
hubs_3, authorities_3 = hits_algorithm(A_3)
print(hubs_1, authorities_1)
print(hubs_2, authorities_2)
print(hubs_3, authorities_3)


# In[ ]:





# In[48]:


import numpy as np
from scipy.linalg import inv, solve


# In[49]:


# codes for question 2.6
def groupInverse(A):
    n = A.shape[0]
    A_2 = A @ A
    rank_A = np.linalg.matrix_rank(A)
    rank_A2 = np.linalg.matrix_rank(A_2)
    if rank_A != rank_A2:
        raise ValueError("can not calculate group inverse because the rank values are diff")

    #Compute group inverse using the pseudoinverse
    inverse = np.linalg.pinv(A)
    B = A @ inverse
    C = inverse @ (np.eye(n) - B)
    group_inverse = B @ A @ C
    return group_inverse

# test 
A = np.array([
    [0.0,        0.33333333, 0.33333333, 0.33333333, 0.0,        0.0],
    [0.0,        0.0,        0.0,        0.5,        0.5,        0.0],
    [0.0,        0.0,        0.0,        0.5,        0.0,        0.5],
    [0.0,        0.0,        0.0,        0.0,        0.0,        1.0],
    [0.0,        0.0,        0.0,        0.5,        0.0,        0.5],
    [0.5,        0.5,        0.0,        0.0,        0.0,        0.0]
])
print(groupInverse(A))


# In[53]:


def find__highest_sensitive_link(A, v):
    n = A.shape[0]
    I = np.eye(n)
    A0 = I - A
    # using above method to get group inverse
    A0_group_inv = groupInverse(A0)
    max_sensitivity = 0
    most_sensitive_link = (None, None)
    for i in range(n):
        for j in range(n):
            if i != j:  
                F = np.zeros((n, n))
                F[i, j] = 1  
                # provides fomula
                d_pi_div_dt = v @ F @ A0_group_inv @ np.linalg.inv(I - 0 * F @ A0_group_inv)
                sensitivity = np.linalg.norm(d_pi_div_dt)
                
                # Update the most sensitive link if current sensitivity is greater
                if sensitivity > max_sensitivity:
                    max_sensitivity = sensitivity
                    most_sensitive_link = (i, j)

    return most_sensitive_link, max_sensitivity


# In[ ]:




