# TagSynapse Formal Tag Model

This document specifies the formal model underlying **TagSynapse**: its vocabularies, Tag Knowledge Graph, relation semantics, semantic candidate generation, graph learning, collaborative evidence, vocabulary evolution, and relational retrieval.

---

## 1. Objects, Users, and Vocabularies

Let the information objects be:

```math
\mathcal{D}
=
\{d_1,\ldots,d_N\}
```

and users:

```math
\mathcal{U}
=
\{u_1,\ldots,u_M\}
```

The initialized controlled vocabulary is:

```math
\mathcal{C}_0
=
\{c_1,\ldots,c_K\}
```

The user-generated folksonomic vocabulary at time `t` is:

```math
\mathcal{F}_t
=
\{f_1,\ldots,f_{J_t}\}
```

The two vocabularies are distinct:

```math
\mathcal{F}_t
\neq
\mathcal{C}_t
```

A folksonomic tag can therefore participate in the knowledge structure without becoming a controlled concept.

---

## 2. Tag Knowledge Graph

At time `t`:

```math
\mathcal{G}_t
=
(V_t,E_t,\mathcal{R})
```

with:

```math
V_t
=
\mathcal{C}_t
\cup
\mathcal{F}_t
```

Each node retains provenance:

```math
\tau(v)
\in
\{
\text{controlled concept},
\text{folksonomic tag}
\}
```

Relations are partitioned into ontological and empirical families:

```math
\mathcal{R}
=
\mathcal{R}_{\mathrm{ont}}
\cup
\mathcal{R}_{\mathrm{emp}}
```

where:

```math
\mathcal{R}_{\mathrm{ont}}
=
\{
\texttt{broader},
\texttt{narrower},
\texttt{equivalent},
\texttt{associated}
\}
```

and:

```math
\mathcal{R}_{\mathrm{emp}}
=
\{
\texttt{coUsedWith},
\texttt{usedFor},
\texttt{createdBy}
\}
```

Empirical association and ontological assertion remain distinct.

---

## 3. Hierarchical Semantics

Let:

```math
c_i
\prec
c_j
```

mean that `c_i` is narrower than `c_j`.

The inverse relation is:

```math
c_j
\succ
c_i
```

Hierarchical membership is transitive:

```math
c_i \prec c_j
\land
c_j \prec c_k
\Rightarrow
c_i \prec^\ast c_k
```

Associative relevance is represented separately:

```math
A(c_i,c_j)
\in
[0,1]
```

A concept pair can therefore participate in both hierarchical and associative relationships without conflating them.

---

## 4. Semantic Representation

For controlled concept `c`, construct a textual representation:

```math
T(c)
=
\operatorname{prefLabel}(c)
\oplus
\operatorname{altLabels}(c)
\oplus
\operatorname{description}(c)
\oplus
\operatorname{facet}(c)
```

where `⊕` denotes structured concatenation.

A dense encoder maps text into a `p`-dimensional representation:

```math
f_\theta:
\text{text}
\rightarrow
\mathbb{R}^{p}
```

Controlled concepts, information objects, and folksonomic tags are represented as:

```math
\mathbf{x}_c
=
f_\theta(T(c))
```

```math
\mathbf{x}_d
=
f_\theta(d)
```

```math
\mathbf{x}_f
=
f_\theta(f)
```

---

## 5. Semantic Candidate Generation

For input `x` and controlled concept `c`:

```math
s_{\mathrm{sem}}(x,c)
=
\cos(\mathbf{x}_x,\mathbf{x}_c)
```

The semantic candidate set is:

```math
C_{\mathrm{sem}}^{(k)}(x)
=
\operatorname{TopK}_{c \in \mathcal{C}_t}
s_{\mathrm{sem}}(x,c)
```

This performs **candidate generation**, not ontology inference.

Semantic similarity does not establish hierarchy:

```math
\cos(\mathbf{x}_a,\mathbf{x}_b)
\not\Rightarrow
a \prec b
```

Nor does it establish equivalence:

```math
\cos(\mathbf{x}_a,\mathbf{x}_b)
\not\Rightarrow
a \equiv b
```

The conceptual division is therefore:

```text
semantic similarity
        ↓
candidate generation

typed relation inference
+ graph structure
+ usage evidence
        ↓
conceptual relationship
```

---

## 6. Tag Recommendation

For information object `d`, define the candidate pool:

```math
C(d)
=
C_{\mathrm{sem}}(d)
\cup
C_{\mathrm{graph}}(d)
\cup
C_{\mathrm{folk}}(d)
```

where:

- `C_sem(d)` contains semantic candidates;
- `C_graph(d)` contains graph-mediated candidates;
- `C_folk(d)` contains established folksonomic tags.

The user selects:

```math
T_d
\subseteq
C(d)
\cup
T_{\mathrm{new}}
```

Equivalently:

```math
T_d
=
H(d,C(d))
```

where `H` denotes the human annotation decision.

---

## 7. Collaborative Usage

Each accepted annotation can be represented as an event:

```math
e
=
(u,d,t,\tau_e)
```

containing user, information object, tag, and timestamp.

Define assignment incidence:

```math
Y_{dt}
=
\begin{cases}
1 & \text{if tag } t \text{ is assigned to object } d \\
0 & \text{otherwise}
\end{cases}
```

Tag co-use count is:

```math
N_{ij}
=
\sum_d
Y_{di}Y_{dj}
```

A normalized co-use measure can use Jaccard association:

```math
J(i,j)
=
\frac{N_{ij}}
{N_i+N_j-N_{ij}}
```

or pointwise mutual information:

```math
\operatorname{PMI}(i,j)
=
\log
\frac{P(i,j)}
{P(i)P(j)}
```

This produces empirical relations such as:

```math
(i,\texttt{coUsedWith},j,w_{ij})
```

---

## 8. Graph Representation Learning

Each node begins with its semantic representation:

```math
\mathbf{h}_v^{(0)}
=
\mathbf{x}_v
```

A relation-aware message-passing formulation is:

```math
\mathbf{h}_i^{(\ell+1)}
=
\sigma
\left(
W_0^{(\ell)}
\mathbf{h}_i^{(\ell)}
+
\sum_{r \in \mathcal{R}}
\sum_{j \in \mathcal{N}_i^r}
\frac{1}{c_{i,r}}
W_r^{(\ell)}
\mathbf{h}_j^{(\ell)}
\right)
```

After `L` layers:

```math
\mathbf{h}_i
=
\mathbf{h}_i^{(L)}
```

Thus:

- `x_i` represents textual semantics;
- `h_i` represents textual semantics conditioned on graph context.

---

## 9. Typed Link Prediction

For graph triple `(i,r,j)`, define:

```math
s_r(i,j)
=
\mathbf{h}_i^\top
R_r
\mathbf{h}_j
```

Relation probability is:

```math
P(r \mid i,j)
=
\sigma(s_r(i,j))
```

For observed positive edges `E+` and corrupted negatives `E-`, a link-prediction objective is:

```math
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
```

This supplies the general formal basis for relation-aware graph learning.

---

## 10. Relation Inference for New Tags

For new folksonomic tag `f`, first retrieve candidate anchors:

```math
A_f
=
C_{\mathrm{sem}}^{(k)}(f)
```

For each candidate concept `c`, infer a typed relation:

```math
g_\phi(f,c)
\rightarrow
P(r \mid f,c)
```

where:

```math
r
\in
\{
\text{narrower},
\text{broader},
\text{equivalent},
\text{associated},
\text{unrelated}
\}
```

The preferred relation is:

```math
\hat{r}
=
\arg\max_r
P(r \mid f,c)
```

Semantic retrieval narrows the candidate space; relation inference determines conceptual placement.

---

## 11. Behavioral Subsumption

Collaborative use can provide directional evidence.

For tags `x` and `y`:

```math
P(y \mid x)
=
\frac{
\sum_d Y_{dx}Y_{dy}
}{
\sum_d Y_{dx}
}
```

Define asymmetry:

```math
B(x,y)
=
P(y \mid x)
-
P(x \mid y)
```

If:

```math
B(x,y)
\gg
0
```

then `x` occurs primarily in contexts occupied by `y`, providing evidence that `x` may be narrower than `y`.

Conversely:

```math
B(x,y)
\ll
0
```

provides evidence in the opposite direction.

Behavioral subsumption is evidence rather than proof.

---

## 12. Structural Plausibility

For proposed placement `(f,r,c)`, let:

```math
S_{\mathrm{graph}}(f,c)
```

measure compatibility with the current graph.

Structural evidence may include:

- graph-model link score;
- neighborhood similarity;
- hierarchy depth;
- parent-child consistency;
- redundant-ancestor avoidance.

A semantically strong candidate is therefore not automatically the structurally preferred placement.

---

## 13. Structural Constraints

Define:

```math
Q(f,r,c)
\in
\{0,1\}
```

where `Q = 1` denotes a structurally admissible proposal.

A proposed relation is rejected if it introduces:

- a hierarchy cycle;
- inverse inconsistency;
- invalid hierarchy;
- duplicate identity;
- a forbidden self-link.

Constraint checking provides a deterministic validation layer around statistical predictions.

---

## 14. Composite Relation Score

For folksonomic tag `f`, candidate relation `r`, and candidate concept `c`:

```math
S_{\mathrm{rel}}(f,r,c)
=
\alpha
S_{\mathrm{text}}(f,r,c)
+
\beta
S_{\mathrm{GNN}}(f,r,c)
+
\gamma
S_{\mathrm{beh}}(f,r,c)
+
\eta
S_{\mathrm{lex}}(f,r,c)
```

subject to:

```math
Q(f,r,c)
=
1
```

The preferred placement is:

```math
(c^\ast,r^\ast)
=
\arg\max_{c,r}
S_{\mathrm{rel}}(f,r,c)
```

The component weights can be calibrated or learned from validated relationships.

---

## 15. Existing Concept vs. Novel Concept

Define an equivalence score:

```math
E(f,c)
```

and its maximum over the controlled vocabulary:

```math
E^\ast(f)
=
\max_{c \in \mathcal{C}_t}
E(f,c)
```

If:

```math
E^\ast(f)
\ge
\tau_{\mathrm{eq}}
```

the new string may map to an existing concept or become an alternate label.

Otherwise, the candidate may remain folksonomic or be considered for controlled-vocabulary admission:

```text
existing concept
alternate label
folksonomic tag
new controlled concept
reject
```

Folksonomic persistence is therefore a legitimate state rather than failed canonicalization.

---

## 16. Novel-Concept Admission

A novel candidate can be evaluated using multiple evidence dimensions.

### Novelty

```math
N(f)
=
1
-
\max_{c \in \mathcal{C}_t}
\cos(\mathbf{x}_f,\mathbf{x}_c)
```

### Support

For `n_f` uses and `m_f` distinct users:

```math
F(f)
=
\lambda_1
\log(1+n_f)
+
\lambda_2
\log(1+m_f)
```

### Contextual coherence

For documents using `f`:

```math
D_f
=
\{d:Y_{df}=1\}
```

define:

```math
C(f)
=
\frac{2}
{|D_f|(|D_f|-1)}
\sum_{i<j}
\cos(
\mathbf{x}_{d_i},
\mathbf{x}_{d_j}
)
```

### Placement confidence

```math
R(f)
=
\max_{c,r}
S_{\mathrm{rel}}(f,r,c)
```

A general admission function is:

```math
A(f)
=
w_N N(f)
+
w_F F(f)
+
w_C C(f)
+
w_R R(f)
```

with admission when:

```math
A(f)
\ge
\tau_A
```

The weights and threshold are parameters to be calibrated empirically.

---

## 17. Confidence-Based Automation

Let `p_1` denote the highest predicted relation probability and `p_2` the second-highest.

Automatic placement can be permitted when:

```math
p_1
\ge
\tau_H
```

and:

```math
p_1-p_2
\ge
\delta
```

Otherwise, the relation is routed to human validation.

```text
high confidence ──────→ automatic placement
uncertain relation ───→ human review
```

---

## 18. Graph Update

For a validated new controlled concept `f`:

```math
\mathcal{C}_{t+1}
=
\mathcal{C}_t
\cup
\{f\}
```

For accepted relation `(f,r*,c*)`:

```math
E_{t+1}
=
E_t
\cup
\{
(f,r^\ast,c^\ast)
\}
```

Therefore:

```math
\mathcal{G}_{t+1}
=
\mathcal{G}_t
+
\Delta \mathcal{G}_t
```

The graph-update loop is:

```text
G_t
 ↓
recommendations
 ↓
human tagging
 ↓
validated usage and relations
 ↓
G_(t+1)
```

---

## 19. Graph-Enhanced Tag Recommendation

For information object `d`, semantic retrieval provides seed concepts:

```math
S_d
=
C_{\mathrm{sem}}^{(k)}(d)
```

For graph candidate `c`:

```math
s_{\mathrm{graph}}(c \mid d)
=
\sum_{s \in S_d}
w_s
\max_r
P(r \mid s,c)
```

Semantic and graph evidence can be combined:

```math
S_{\mathrm{tag}}(c \mid d)
=
\alpha
s_{\mathrm{sem}}(d,c)
+
(1-\alpha)
s_{\mathrm{graph}}(c \mid d)
```

The graph-mediated candidate set is then:

```math
C_{\mathrm{graph}}(d)
=
\operatorname{TopK}
S_{\mathrm{tag}}(c \mid d)
```

Graph structure can therefore surface candidates that semantic similarity alone does not retrieve.

---

## 20. Relational Retrieval

For query `q`, semantic retrieval produces:

```math
S_q
=
C_{\mathrm{sem}}^{(k)}(q)
```

Graph expansion produces:

```math
G_q
=
\{
c:
s_{\mathrm{graph}}(c \mid S_q)
>
\tau_g
\}
```

The expanded concept representation is:

```math
Q^\ast
=
S_q
\cup
G_q
```

For document tag set `T_d`, define concept compatibility:

```math
K(c,t)
=
\begin{cases}
1 & c=t \\
\kappa_h(c,t) & \text{hierarchically related} \\
\kappa_a(c,t) & \text{associatively related} \\
0 & \text{otherwise}
\end{cases}
```

A graph-enhanced retrieval score is:

```math
S_{\mathrm{ret}}(d \mid q)
=
\lambda
S_{\mathrm{text}}(d,q)
+
(1-\lambda)
\sum_{c \in Q^\ast}
w_c
\max_{t \in T_d}
K(c,t)
```

This combines direct textual relevance with concept-level relational relevance.

---

## 21. Hierarchical Retrieval

For concept `c`, define descendants:

```math
\operatorname{Desc}(c)
=
\{
x:
x \prec^\ast c
\}
```

and ancestors:

```math
\operatorname{Anc}(c)
=
\{
x:
c \prec^\ast x
\}
```

Expansion can be weighted by graph distance:

```math
w(c,x)
=
e^{-\rho d_{\mathcal{G}}(c,x)}
```

Closer hierarchical relatives therefore contribute more strongly than distant ones.

---

## 22. Folksonomic Retrieval

Folksonomic tags remain useful for retrieval before formal promotion.

For query `q`:

```math
S_{\mathrm{folk}}(d,q)
=
\max_{f \in T_d}
\cos(\mathbf{x}_q,\mathbf{x}_f)
```

If a folksonomic tag is connected to a controlled concept, retrieval can traverse the folksonomy–ontology boundary while preserving the original tag.

---

## 23. Experimental Decomposition

The architecture supports three principal assistance conditions.

### `M0` — Manual tagging

```math
M_0
=
\text{manual tagging}
```

### `M1` — Semantic mediation

```math
M_1
=
M_0
+
C_{\mathrm{sem}}
```

### `M2` — Structural-semantic TagSynapse

```math
M_2
=
M_1
+
C_{\mathrm{graph}}
+
\text{relational inference}
```

An adaptive vocabulary-evolution extension can be represented as:

```math
M_3
=
M_2
+
\text{concept admission/evolution}
```

The comparisons isolate:

- `M0 → M1`: contribution of semantic mediation;
- `M1 → M2`: contribution of graph structure;
- `M2 → M3`: contribution of adaptive vocabulary evolution.

---

## 24. Compact System Definition

The complete TagSynapse model can be summarized as:

```math
\mathcal{G}_{t+1}
=
\Phi
\left(
\mathcal{G}_t,
H
\left[
\operatorname{Retrieve}
\left(
\mathcal{D},
f_\theta,
\mathcal{G}_t,
\operatorname{GNN}_\psi
\right)
\right],
\operatorname{Rel}_\phi,
\operatorname{Usage}_t
\right)
```

where:

- `f_θ` provides semantic representation;
- `Retrieve` produces semantic and graph-mediated candidates;
- `H` represents human tagging decisions;
- `Rel_φ` infers typed conceptual relations;
- `Usage_t` captures accumulated collaborative evidence;
- `Φ` validates and updates the graph.

The model can be summarized conceptually as:

> **Users organize information through the graph, and their organization of information changes the graph.**
