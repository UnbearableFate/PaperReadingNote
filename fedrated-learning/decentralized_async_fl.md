# Decentralized Federated Learning

## H. Attiya and N. Schiller, “Asynchronous Fully-Decentralized SGD in the Cluster-Based Model,” vol. 13898, 2023, pp. 52–66. doi: 10.1007/978-3-031-30448-4_5.

Strategy: Collect N non-stale model parameters and aggregate them

## E. T. M. Beltrán et al., “Decentralized Federated Learning: Fundamentals, State of the Art, Frameworks, Trends, and Challenges,” IEEE Commun. Surv. Tutorials, vol. 25, no. 4, pp. 2983–3013, 2023, doi: 10.1109/COMST.2023.3315746.

### FUNDAMENTALS AND TAXONOMY

#### Federation Architecture

- cross-silo DFL
  - usually a relatively small number of nodes (<100), each with a large amount of data (about millions of samples),
  - nodes use consistent, robust, and scalable computing over time.

- cross-device DFL
  - the number of nodes is relatively large, where each node has a relatively small amount of data
  - limited computational power
  - nodes could periodically disconnect from the network so that the network dropout rate would increase considerably

##### Participant Role

trainer, aggregator, proxy, and idle.

##### Decentralization Schema

DFL, Semi-Decentralized Federated Learning (SDFL), and CFL.

- In DFL, participants perform four steps independently: local model training, parameter exchange, local model aggregation, and parameter exchange again.

- In SDFL, participants perform the first two steps, while an aggregator participant handles the third step and transfers leadership for the aggregation functionality (step 5).

- In CFL, a central server handles parameter aggregation (step 3), with the rest of the network receiving and updating their local models accordingly (steps 4 and 5).

##### Data Distribution

- continue aggregating them at a higher level

- accept the presence of different federated models based on the network topology and characteristics of the nodes

#### Network Topology

- Fully Connected Networks
  - cost is high
  - adding new nodes increases the complexity of managing connections for each node
  - this topology is highly reliable and robust
- Partially Connected Networks
  - star-structured
  - ring-structured
  - random
    - Erdös–Rényi model
- Node Clustering

- Communications Scheme:
  - synchronous communication
  - asynchronous communications
  - semi-synchronous communications
- Peer-to-Peer
  - The main drawback of DFL is that nodes cannot randomly choose d neighbors among existing nodes since there is no central coordinator to determine these associations
- Gossip Communications:

#### Key Performance Indicators

- Federation Nodes KPIs:
  - resource capabilities and node mobility are identified as promising KPIs to determine node performance.
    - Computing capacity
    - Network capacity.
- Federation Communications KPIs
might experience instability due to asynchronous communications, leading to nodes processing and exchanging data at varying rates.
result in delays and inconsistencies in the learning process
  - **Federation size**
    - Reduced network size improves parameter transmission performance by reducing interference
  - Robustness of communication links.
- **Federation Models KPIs**
  - Model capacity
    - model loss, accuracy, sensitivity, specificity, and the time associated with the convergence of the model.

#### DFL Optimizations

- selection of nodes
  - Sequential selection of nodes.
  - Random selection of nodes.
  - Scheme schedule selection of nodes.
- Customizations of FedAvg and new techniques
  - Decentralized Stochastic Gradient Descent (DSGD).
  - `FedPGA` (Reading)
  - Dynamic Average Consensus-based FL (DACFL).
  - Split Learning
  - Decentralized FedAvg with Momentum
  - DeceFL
- Optimization of Federation Communications
  - different distributed optimization schemes aim to maintain acceptable convergence rates in terms of recurring iterations and device computation time.
    - decentralized Alternating Direction Method of Multipliers (ADMM) [105], EXTRA [106], and ADMM based on Jacobi-Proximal [107].
  - use an incremental learning method to reduce costs by activating and linking agents while keeping other nodes and links inactive.
    - These include Random Walk ADMM (WADMM) [134], Parallel Random Walk ADMM (PW-ADMM) [123] and Walk Proximal Gradient (WPG) [135] which are commonly used in increment-based approaches.
  - optimize the balance between improving the quality of the model and saving communication resources,
    - These mechanisms typically offer an **independent selection of participants and fragments of the NN** to be transmitted, providing a promising alternative to traditional optimizations.
- Regarding optimizing network overhead
  - quantization and sparsification
- Optimization of Federation Models:

#### TRENDS, LESSONS LEARNED, AND OPEN  CHALLENGES

##### TRENDS

- Federation architectures, network topologies, and communication mechanisms are extensively studied.
- Fully connected network topologies are widely applied to DFL scenarios 
  - about 50% of the papers analyzed in the network topology fundamental belong to this approach.
- The optimization of communications is predominant in recent work on DFL.

##### Lessons Learned

- The use of specific aggregation algorithms for DFL is still limited.
- There is a limited number of solutions providing realistic federation benchmarks.
- There is no consensus on frameworks in the literature for deploying DFL architectures.
- There is a lack of literature using unsupervised learning in DFL architectures.

##### Open Challenges

- Improve the scalability of the solution when the number of participants in the federation increases.
- Ensure the homogeneous participation of the constrained nodes in the federation.
- Address participant mobility.
- Create modular, scalable, and efficient frameworks for diverse application scenarios.
- Handle heterogeneous datasets in decentralized participants.
- Adapt the dynamic scheduling of the federated network to the application scenario.
- Explore standardization activities for DFL.

#### Conclusion

