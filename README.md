# Duoploy-Ridesourcing-market

Over the past decade, ride-hailing services have reshaped how people travel in cities and substantially influenced the traditional taxi industry and transport systems. Multiple platforms co-exist and compete with each other in many local ride-hailing markets. These platforms compete not only on the demand side for passengers but also on the supply side for freelance drivers who may work for multiple platforms. In this paper, we consider a price and service time sensitive market in which two ride-hailing platforms provide substitutable services. They compete for the same group of passengers and drivers, and they decide the matching, order fare and driver wage dynamically and independently. We then examine how to win the inter-platform competition in this duopoly ride-hailing market from the perspective of a single platform. The competition is formulated as a joint matching, fare, and wage optimization problem for the controlled platform under three cases: (i) Observable static (OS) competition, (ii) Non-observable static (NOS) competition, and (iii) Non-observable dynamic (NOD) competition. We show that the proposed model helps the controlled platform to take a dominant market share and creates substantial benefits for the drivers and the passengers.

## Three competition games

The competition problem falls into the category of a repeated game with an infinite number of periods (i.e.,~matching time window). At the end of each period, platform $i$ may rely on the previously observed information when it chooses the joint strategy (action) for the next repetition. Hence, we analyze the problem in three different settings,

- **Observable static (OS) competition**, in which platform $i$ is assumed to have complete visibility (i.e.,~access to perfect information) of platform $j$ at every time step, and the decision-making of platform $j$ is irrelevant with that of platform $i$. Although inapplicable in a real-world competition, this setting would lay the methodological foundation for solving other realistic settings.
- **Non-observable static (NOS) competition**, in which platform $i$ has no knowledge of the decision made by platform $j$ before making its own decision, and the decision-making of platform $j$ is irrelevant with that of platform $i$.
- **Non-observable dynamic (NOD) competition**, in which platform $i$ does not know the decision made by platform $j$ before making its own decision, while platform $j$ is an intelligent player and changes its strategy according to market dynamics in real-time. 


## Observable static competition

The OS competition assumes fare, waging, and matching decisions  $\theta_{0}$, $\theta_{1}$, $\theta_{2}$, $\lambda$, and $\kappa$ are completely available to platform 1 at each time step. Accordingly, $f_{o,d,2}$, $w_{o,d,2}$, and $x_{o,d,2}$ are also completely visible to platform 1.

We presented a humegenuous case in this ([Example of OS Competition](https://github.com/marco-yue/Duoploy-Ridesourcing-Competition/blob/main/01%20Example%20(OS%20Competition).ipynb)).

## Nonobservable static competition

The strategy taken by platform 2 is not available for platform 1 in the NOS competition. Instead of finding provably optimal strategies in the OS competition, the NOS competition requires a different model and approach. The problem of NOS competition can be modelled as a continuum-armed bandit (CAB) problem \citep{agrawal1995continuum}. We assume that an arm applied in the CAB problem is a quadruple strategy set taken by platform 2,

$$
    a_{n}=\{\hat{\theta}_{0,n},\hat{\theta}_{1,n},\hat{\theta}_{2,n},\hat{\lambda}_{n}\}
$$

We presented a humegenuous case in this ([Example of NOS Competition](https://github.com/marco-yue/Duoploy-Ridesourcing-Competition/blob/main/02%20Example%20(NOS%20Competition).ipynb)).

