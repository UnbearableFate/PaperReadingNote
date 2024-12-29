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
- 