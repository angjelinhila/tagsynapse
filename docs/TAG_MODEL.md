# TagSynapse Formal Tag Model

This document specifies the formal model underlying TagSynapse: its tag representation, knowledge graph, relation 
semantics, semantic candidate generation, graph learning, vocabulary evolution, and relational retrieval.

---

## 1. Notation

Let:

$$
\mathcal{D} = \{d_1,\dots,d_N\}
$$

denote information objects, and:

$$
\mathcal{U} = \{u_1,\dots,u_M\}
$$

denote users.

The controlled vocabulary at initialization is:

$$
\mathcal{C}_0 = \{c_1,\dots,c_K\}
$$

The folksonomic vocabulary at time $t$ is:

$$
\mathcal{F}_t = \{f_1,\dots,f_{J_t}\}
$$

The two vocabularies are distinct:

$$
\mathcal{F}_t \neq \mathcal{C}_t
$$

A user-generated tag can therefore participate in TagSynapse without becoming a controlled concept.

---

## 2. Tag Knowledge Graph

At time $t$:

$$
\mathcal{G}_t = (V_t,E_t,\mathcal{R})
$$

with:

$$
V_t = \mathcal{C}_t \cup \mathcal{F}_t
$$

Each node retains provenance:

$$
\tau(v)
\in
\{
\text{controlled concept},
\text{folksonomic tag}
\}
$$

Relations are partitioned into two families:

$$
\mathcal{R}
=
\mathcal{R}_{\mathrm{ont}}
\cup
\mathcal{R}_{\mathrm{emp}}
$$

### Ontological relations

$$
\mathcal{R}_{\mathrm{ont}}
=
\{
\texttt{broader},
\texttt{narrower},
\texttt{equivalent},
\texttt{associated}
\}
$$

### Empirical relations

$$
\mathcal{R}_{\mathrm{emp}}
=
\{
\texttt{coUsedWith},
\texttt{usedFor},
\texttt{createdBy}
\}
$$

Empirical association and ontological assertion are not interchangeable.

---

## 3. Hierarchical Semantics

Let:

$$
c_i \prec c_j
$$

mean that $c_i$ is narrower than $c_j$.

The inverse relation is:

$$
c_j \succ c_i
$$

Hierarchy is transitive:

$$
c_i \prec c_j
\land
c_j \prec c_k
\Rightarrow
c_i \prec^\ast c_k
$$

Associative relevance is represented separately:

$$
A(c_i,c_j) \in [0,1]
$$

A pair may therefore have both hierarchical and associative relationships without conflating the two.

---

## 4. Semantic Representations

For controlled concept $c$, define a textual representation:

$$
T(c)
=
\operatorname{prefLabel}(c)
\oplus
\operatorname{altLabels}(c)
\oplus
\operatorname{description}(c)
\oplus
\operatorname{facet}(c)
$$

where $\oplus$ denotes structured concatenation.

A dense encoder:

$$
f_\theta :
\text{text}
\rightarrow
\mathbb{R}^p
$$

produces:

$$
\mathbf{x}_c = f_\theta(T(c))
$$

for controlled concepts,

$$
\mathbf{x}_d = f_\theta(d)
$$

for information objects, and:

$$
\mathbf{x}_f = f_\theta(f)
$$

for folksonomic tags.

---

## 5. Semantic Candidate Generation

For input $x$ and controlled concept $c$:

$$
s_{\mathrm{sem}}(x,c)
=
\cos(\mathbf{x}_x,\mathbf{x}_c)
$$

The semantic candidate set is:

$$
C_{\mathrm{sem}}^{(k)}(x)
=
\operatorname{TopK}_{c \in \mathcal{C}_t}
s_{\mathrm{sem}}(x,c)
$$

This is a candidate-generation mechanism.

Semantic similarity does not establish hierarchical direction:

$$
\cos(\mathbf{x}_a,\mathbf{x}_b)
\not\Rightarrow
a \prec b
$$

Nor does it establish equivalence:

$$
\cos(\mathbf{x}_a,\mathbf{x}_b)
\not\Rightarrow
a \equiv b
$$

The conceptual division is therefore:

```text
semantic similarity
        ↓
candidate generation

relation inference + graph structure + usage
        ↓
conceptual relationship
```

---

## 6. Tag Recommendation

For information object $d$, define:

$$
C(d)
=
C_{\mathrm{sem}}(d)
\cup
C_{\mathrm{graph}}(d)
\cup
C_{\mathrm{folk}}(d)
$$

where:

- $C_{\mathrm{sem}}$ contains semantic candidates;
- $C_{\mathrm{graph}}$ contains graph-mediated candidates;
- $C_{\mathrm{folk}}$ contains established folksonomic tags.

The user selects:

$$
T_d
\subseteq
C(d)
\cup
T_{\mathrm{new}}
$$

Equivalently:

$$
T_d = H(d,C(d))
$$

where $H$ represents the human annotation process.

---

## 7. Collaborative Usage

Each accepted annotation is an event:

$$
e = (u,d,t,\tau_e)
$$

containing user, information object, tag, and timestamp.

Define:

$$
Y_{dt}
=
\begin{cases}
1 & \text{if tag } t \text{ is assigned to } d \\
0 & \text{otherwise}
\end{cases}
$$

Tag co-use count is:

$$
N_{ij}
=
\sum_d Y_{di}Y_{dj}
$$

A normalized association can use Jaccard similarity:

$$
J(i,j)
=
\frac{N_{ij}}
{N_i+N_j-N_{ij}}
$$

or pointwise mutual information:

$$
\operatorname{PMI}(i,j)
=
\log
\frac{P(i,j)}
{P(i)P(j)}
$$

This produces empirical edges such as:

$$
(i,\texttt{coUsedWith},j,w_{ij})
$$

---

## 8. GNN Representation Learning

Initial node state is:

$$
\mathbf{h}_v^{(0)}
=
\mathbf{x}_v
$$

A relation-aware message-passing layer can be written as:

$$
\mathbf{h}_i^{(\ell+1)}
=
\sigma
\left(
W_0^{(\ell)}\mathbf{h}_i^{(\ell)}
+
\sum_{r \in \mathcal{R}}
\sum_{j \in \mathcal{N}_i^r}
\frac{1}{c_{i,r}}
W_r^{(\ell)}
\mathbf{h}_j^{(\ell)}
\right)
$$

After $L$ layers:

$$
\mathbf{h}_i
=
\mathbf{h}_i^{(L)}
$$

Thus:

- $\mathbf{x}_i$ represents textual semantics;
- $\mathbf{h}_i$ represents semantics conditioned on graph structure.

---

## 9. Typed Link Prediction

For graph triple $(i,r,j)$, define:

$$
s_r(i,j)
=
\mathbf{h}_i^\top
R_r
\mathbf{h}_j
$$

Relation probability is:

$$
P(r \mid i,j)
=
\sigma(s_r(i,j))
$$

For positive edges $E^+$ and negative edges $E^-$:

$$
\mathcal{L}_{\mathrm{link}}
=
-
\sum_{(i,r,j)\in E^+}
\log \sigma(s_r(i,j))
-
\sum_{(i,r,j)\in E^-}
\log
\left(
1-\sigma(s_r(i,j))
\right)
$$

This provides the formal basis for typed relation prediction.

---

## 10. Relation Inference for New Tags

For new folksonomic tag $f$, retrieve candidate anchors:

$$
A_f
=
C_{\mathrm{sem}}^{(k)}(f)
$$

For each candidate concept $c$:

$$
g_\phi(f,c)
\rightarrow
P(r \mid f,c)
$$

where:

$$
r
\in
\{
\text{narrower},
\text{broader},
\text{equivalent},
\text{associated},
\text{unrelated}
\}
$$

The preferred relation is:

$$
\hat{r}
=
\arg\max_r
P(r \mid f,c)
$$

Semantic retrieval narrows the candidate space; relation inference determines conceptual placement.

---

## 11. Behavioral Subsumption

Usage can provide directional evidence.

For tags $x$ and $y$:

$$
P(y \mid x)
=
\frac{
\sum_dY_{dx}Y_{dy}
}{
\sum_dY_{dx}
}
$$

Define:

$$
B(x,y)
=
P(y \mid x)
-
P(x \mid y)
$$

If:

$$
B(x,y) \gg 0
$$

then $x$ occurs primarily within contexts occupied by $y$, providing evidence that $x$ may be narrower than $y$.

Conversely:

$$
B(x,y) \ll 0
$$

provides evidence in the opposite direction.

Behavioral subsumption is evidence rather than proof.

---

## 12. Structural Plausibility

For proposed relation $(f,r,c)$, let:

$$
S_{\mathrm{graph}}(f,c)
$$

measure compatibility with the existing graph.

Potential components include:

- GNN link score;
- neighborhood similarity;
- hierarchy depth;
- parent-child consistency;
- redundant ancestor avoidance.

The strongest semantic relation is not necessarily the structurally best placement.

---

## 13. Structural Constraints

Define:

$$
Q(f,r,c)
\in
\{0,1\}
$$

where $Q=1$ indicates a structurally admissible relation.

A proposed edge is rejected when it introduces:

- cycles;
- inverse inconsistency;
- invalid hierarchy;
- duplicate identity;
- self-links.

Graph constraints therefore provide a deterministic validation layer around statistical predictions.

---

## 14. Composite Relation Score

For candidate folksonomic tag $f$, relation $r$, and controlled concept $c$:

$$
S_{\mathrm{rel}}(f,r,c)
=
\alpha S_{\mathrm{text}}(f,r,c)
+
\beta S_{\mathrm{GNN}}(f,r,c)
+
\gamma S_{\mathrm{beh}}(f,r,c)
+
\eta S_{\mathrm{lex}}(f,r,c)
$$

subject to:

$$
Q(f,r,c)=1
$$

The preferred placement is:

$$
(c^\ast,r^\ast)
=
\arg\max_{c,r}
S_{\mathrm{rel}}(f,r,c)
$$

The component weights may be learned from validated relations.

---

## 15. Existing Concept vs. Novel Concept

Define an equivalence score:

$$
E(f,c)
$$

and:

$$
E^\ast(f)
=
\max_{c\in\mathcal{C}_t}
E(f,c)
$$

If:

$$
E^\ast(f)
\ge
\tau_{\mathrm{eq}}
$$

then $f$ may map to an existing concept or become an alternate label.

Otherwise:

$$
f
\rightarrow
\begin{cases}
\text{folksonomic tag} \\
\text{new controlled concept} \\
\text{reject}
\end{cases}
$$

Folksonomic persistence is therefore a legitimate intermediate state.

---

## 16. Novel Concept Admission

A candidate concept can be evaluated using several dimensions.

### Novelty

$$
N(f)
=
1-
\max_{c\in\mathcal{C}_t}
\cos(\mathbf{x}_f,\mathbf{x}_c)
$$

### Support

For $n_f$ uses and $m_f$ distinct users:

$$
F(f)
=
\lambda_1\log(1+n_f)
+
\lambda_2\log(1+m_f)
$$

### Contextual coherence

For:

$$
D_f
=
\{d:Y_{df}=1\}
$$

define:

$$
C(f)
=
\frac{2}
{|D_f|(|D_f|-1)}
\sum_{i<j}
\cos(\mathbf{x}_{d_i},\mathbf{x}_{d_j})
$$

### Placement confidence

$$
R(f)
=
\max_{c,r}
S_{\mathrm{rel}}(f,r,c)
$$

A general admission model is:

$$
A(f)
=
w_NN(f)
+
w_FF(f)
+
w_CC(f)
+
w_RR(f)
$$

Promotion occurs when:

$$
A(f)
\ge
\tau_A
$$

The admission model can be empirically calibrated.

---

## 17. Confidence-Based Automation

Let $p_1$ be the highest predicted relation probability and $p_2$ the second-highest.

Automatic placement is permitted when:

$$
p_1 \ge \tau_H
$$

and:

$$
p_1-p_2 \ge \delta
$$

Otherwise, the candidate is routed to human validation.

This yields:

```text
high confidence
      ↓
automatic placement

uncertain prediction
      ↓
human review
```

---

## 18. Graph Update

For validated new controlled concept $f$:

$$
\mathcal{C}_{t+1}
=
\mathcal{C}_t
\cup
\{f\}
$$

For accepted relation $(f,r^\ast,c^\ast)$:

$$
E_{t+1}
=
E_t
\cup
\{
(f,r^\ast,c^\ast)
\}
$$

Therefore:

$$
\mathcal{G}_{t+1}
=
\mathcal{G}_t
+
\Delta\mathcal{G}_t
$$

The feedback process is:

```text
G_t
 ↓
recommendations
 ↓
human tagging
 ↓
validated knowledge
 ↓
G_(t+1)
```

---

## 19. Graph-Enhanced Tag Recommendation

For information object $d$, semantic retrieval produces seed concepts:

$$
S_d
=
C_{\mathrm{sem}}^{(k)}(d)
$$

For graph candidate $c$:

$$
s_{\mathrm{graph}}(c \mid d)
=
\sum_{s\in S_d}
w_s
\max_r
P(r \mid s,c)
$$

Semantic and graph relevance can be combined:

$$
S_{\mathrm{tag}}(c \mid d)
=
\alpha
s_{\mathrm{sem}}(d,c)
+
(1-\alpha)
s_{\mathrm{graph}}(c \mid d)
$$

Then:

$$
C_{\mathrm{graph}}(d)
=
\operatorname{TopK}
S_{\mathrm{tag}}(c \mid d)
$$

Graph structure can therefore surface candidate tags that semantic similarity alone would not retrieve.

---

## 20. Relational Retrieval

For query $q$, semantic retrieval produces:

$$
S_q
=
C_{\mathrm{sem}}^{(k)}(q)
$$

Graph expansion produces:

$$
G_q
=
\{
c:
s_{\mathrm{graph}}(c\mid S_q)
>
\tau_g
\}
$$

The expanded concept representation is:

$$
Q^\ast
=
S_q
\cup
G_q
$$

For document tag set $T_d$, define compatibility:

$$
K(c,t)
=
\begin{cases}
1 & c=t \\
\kappa_h(c,t) & \text{hierarchically related} \\
\kappa_a(c,t) & \text{associatively related} \\
0 & \text{otherwise}
\end{cases}
$$

Then:

$$
S_{\mathrm{ret}}(d\mid q)
=
\lambda
S_{\mathrm{text}}(d,q)
+
(1-\lambda)
\sum_{c\in Q^\ast}
w_c
\max_{t\in T_d}
K(c,t)
$$

This combines direct textual relevance with concept-level relational relevance.

---

## 21. Hierarchical Retrieval

For concept $c$:

$$
\operatorname{Desc}(c)
=
\{
x:
x\prec^\ast c
\}
$$

and:

$$
\operatorname{Anc}(c)
=
\{
x:
c\prec^\ast x
\}
$$

Expansion can be weighted by graph distance:

$$
w(c,x)
=
e^{-\rho d_{\mathcal{G}}(c,x)}
$$

Closer hierarchical relatives therefore contribute more strongly than distant ones.

---

## 22. Folksonomic Retrieval

Folksonomic tags remain useful for retrieval before formal promotion.

For query $q$:

$$
S_{\mathrm{folk}}(d,q)
=
\max_{f\in T_d}
\cos(\mathbf{x}_q,\mathbf{x}_f)
$$

If folksonomic tag $f$ has a relationship to controlled concept $c$, retrieval can cross the folksonomy–ontology 
boundary.

This allows emergent vocabulary to participate in retrieval immediately.

---

## 23. Experimental Conditions

The architecture naturally supports three primary experimental variants.

### Manual tagging

$$
M_0
=
\text{manual tagging}
$$

### Semantic recommendation

$$
M_1
=
M_0
+
C_{\mathrm{sem}}
$$

### Structural-semantic TagSynapse

$$
M_2
=
M_1
+
C_{\mathrm{graph}}
+
\text{relation inference}
$$

An adaptive vocabulary-evolution condition can additionally be represented as:

$$
M_3
=
M_2
+
\text{concept admission}
$$

This separates the contribution of:

1. semantic mediation;
2. graph structure;
3. adaptive vocabulary evolution.

---

## 24. Compact System Definition

The full TagSynapse process can be summarized as:

$$
\mathcal{G}_{t+1}
=
\Phi
\left(
\mathcal{G}_t,
H[
\operatorname{Retrieve}
(
D,
f_\theta,
\mathcal{G}_t,
\operatorname{GNN}_\psi
)
],
\operatorname{Rel}_\phi,
\operatorname{Usage}_t
\right)
$$

where:

- $f_\theta$ provides semantic representation;
- $\operatorname{Retrieve}$ produces semantic and graph-mediated candidates;
- $H$ represents human tagging decisions;
- $\operatorname{Rel}_\phi$ infers typed conceptual relations;
- $\operatorname{Usage}_t$ represents accumulated collaborative evidence;
- $\Phi$ validates and updates the graph.

The model can be summarized conceptually as:

> **Users organize information through the graph, and their organization of information changes the graph.**
