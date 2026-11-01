# TagSynapse Formal Tag Model

This document specifies the representation, relation model, recommendation process, graph learning, graph evolution, and 
retrieval formalism underlying TagSynapse.

---

## 1. Objects, Users, and Tags

Let:

\[
\mathcal D=\{d_1,\dots,d_N\}
\]

denote information objects and:

\[
\mathcal U=\{u_1,\dots,u_M\}
\]

denote users.

The initialized controlled vocabulary is:

\[
\mathcal C_0=\{c_1,\dots,c_K\}.
\]

The user-generated folksonomic vocabulary at time \(t\) is:

\[
\mathcal F_t=\{f_1,\dots,f_{J_t}\}.
\]

Importantly:

\[
\mathcal F_t\neq\mathcal C_t.
\]

A folksonomic tag may participate in the knowledge graph without becoming a controlled concept.

---

## 2. Tag Knowledge Graph

At time \(t\):

\[
\mathcal G_t=(V_t,E_t,\mathcal R)
\]

where:

\[
V_t=\mathcal C_t\cup\mathcal F_t.
\]

Node provenance is represented by:

\[
\tau(v)\in
\{
\text{controlled concept},
\text{folksonomic tag}
\}.
\]

Relations are partitioned:

\[
\mathcal R
=
\mathcal R_{\mathrm{ont}}
\cup
\mathcal R_{\mathrm{emp}}.
\]

Ontological relations include:

\[
\mathcal R_{\mathrm{ont}}
=
\{
\texttt{broader},
\texttt{narrower},
\texttt{equivalent},
\texttt{associated}
\}.
\]

Empirical relations include:

\[
\mathcal R_{\mathrm{emp}}
=
\{
\texttt{coUsedWith},
\texttt{usedFor},
\texttt{createdBy}
\}.
\]

The two relation classes must remain semantically distinct.

---

## 3. Hierarchy

Let:

\[
c_i\prec c_j
\]

mean that \(c_i\) is narrower than \(c_j\).

The inverse is:

\[
c_j\succ c_i.
\]

Hierarchy is transitive:

\[
c_i\prec c_j
\land
c_j\prec c_k
\Rightarrow
c_i\prec^\ast c_k.
\]

Associative relevance is modeled separately:

\[
A(c_i,c_j)\in[0,1].
\]

Thus two concepts may be hierarchically related and associatively related without conflating those relation types.

---

## 4. Semantic Representations

A controlled concept can be represented textually as:

\[
T(c)
=
\operatorname{prefLabel}(c)
\oplus
\operatorname{altLabels}(c)
\oplus
\operatorname{description}(c)
\oplus
\operatorname{facet}(c).
\]

A dense encoder:

\[
f_\theta:\text{text}\rightarrow\mathbb R^p
\]

produces:

\[
\mathbf x_c=f_\theta(T(c)),
\]

\[
\mathbf x_d=f_\theta(d),
\]

and:

\[
\mathbf x_f=f_\theta(f).
\]

---

## 5. Semantic Candidate Retrieval

Similarity is:

\[
s_{\mathrm{sem}}(x,c)
=
\cos(\mathbf x_x,\mathbf x_c).
\]

The semantic candidate set is:

\[
C_{\mathrm{sem}}^{(k)}(x)
=
\operatorname{TopK}_{c\in\mathcal C_t}
s_{\mathrm{sem}}(x,c).
\]

This is candidate generation, not ontology inference.

\[
\boxed{
\text{semantic retrieval identifies where to look}
}
\]

rather than:

\[
\boxed{
\text{semantic retrieval determines conceptual relation}
}
\]

In particular:

\[
\cos(\mathbf x_a,\mathbf x_b)
\not\Rightarrow
a\prec b
\]

and:

\[
\cos(\mathbf x_a,\mathbf x_b)
\not\Rightarrow
a\equiv b.
\]

---

## 6. Tagging an Information Object

For document \(d\), the recommendation set is:

\[
C(d)
=
C_{\mathrm{sem}}(d)
\cup
C_{\mathrm{graph}}(d)
\cup
C_{\mathrm{folk}}(d).
\]

The user chooses:

\[
T_d
\subseteq
C(d)\cup T_{\mathrm{new}}.
\]

Equivalently:

\[
T_d
=
H(d,C(d)),
\]

where \(H\) represents human annotation decisions.

---

## 7. Collaborative Usage

Each accepted annotation is an event:

\[
e=(u,d,t,\tau_e).
\]

Define:

\[
Y_{dt}
=
\begin{cases}
1 & \text{if tag }t\text{ is assigned to }d\\
0 & \text{otherwise.}
\end{cases}
\]

Co-use count is:

\[
N_{ij}
=
\sum_dY_{di}Y_{dj}.
\]

A normalized association may use Jaccard:

\[
J(i,j)
=
\frac{N_{ij}}
{N_i+N_j-N_{ij}}
\]

or PMI:

\[
\operatorname{PMI}(i,j)
=
\log
\frac{P(i,j)}
{P(i)P(j)}.
\]

This produces empirical edges:

\[
(i,\texttt{coUsedWith},j,w_{ij}).
\]

Observed association remains distinct from ontological hierarchy.

---

## 8. GNN Representation Learning

Initial node state is:

\[
\mathbf h_v^{(0)}
=
\mathbf x_v.
\]

A relational message-passing formulation is:

\[
\mathbf h_i^{(\ell+1)}
=
\sigma
\left[
W_0^{(\ell)}\mathbf h_i^{(\ell)}
+
\sum_{r\in\mathcal R}
\sum_{j\in\mathcal N_i^r}
\frac{1}{c_{i,r}}
W_r^{(\ell)}
\mathbf h_j^{(\ell)}
\right].
\]

After \(L\) layers:

\[
\mathbf h_i
=
\mathbf h_i^{(L)}.
\]

Thus:

\[
\mathbf x_i
\]

captures textual semantics, whereas:

\[
\mathbf h_i
\]

captures textual semantics conditioned on graph context.

---

## 9. Typed Link Prediction

For triple:

\[
(i,r,j),
\]

define relation score:

\[
s_r(i,j)
=
\mathbf h_i^\top R_r\mathbf h_j.
\]

Then:

\[
P(r\mid i,j)
=
\sigma(s_r(i,j)).
\]

For observed positive edges \(E^+\) and corrupted negatives \(E^-\):

\[
\mathcal L_{\mathrm{link}}
=
-
\sum_{(i,r,j)\in E^+}
\log\sigma(s_r(i,j))
-
\sum_{(i,r,j)\in E^-}
\log
\left[
1-\sigma(s_r(i,j))
\right].
\]

This provides the basis for relation-aware graph learning.

---

## 10. Relation Inference for New Tags

For a newly created tag \(f\), first retrieve candidate anchors:

\[
A_f
=
C_{\mathrm{sem}}^{(k)}(f).
\]

For each candidate \(c\), classify:

\[
g_\phi(f,c)
\rightarrow
P(r\mid f,c),
\]

with:

\[
r
\in
\{
\text{narrower},
\text{broader},
\text{equivalent},
\text{associated},
\text{unrelated}
\}.
\]

The predicted relation is:

\[
\hat r
=
\arg\max_rP(r\mid f,c).
\]

Semantic retrieval narrows the search space; directional relation inference determines placement.

---

## 11. Behavioral Subsumption

Collaborative use provides directional evidence.

For tags \(x\) and \(y\):

\[
P(y\mid x)
=
\frac{
\sum_dY_{dx}Y_{dy}
}{
\sum_dY_{dx}
}.
\]

Define:

\[
B(x,y)
=
P(y\mid x)-P(x\mid y).
\]

Then:

\[
B(x,y)\gg0
\]

provides evidence that \(x\) may be narrower than \(y\), while:

\[
B(x,y)\ll0
\]

provides evidence in the opposite direction.

This remains evidence rather than proof.

---

## 12. Structural Plausibility

For proposed placement:

\[
(f,r,c),
\]

let:

\[
S_{\mathrm{graph}}(f,c)
\]

measure compatibility with the existing graph.

Relevant signals may include:

- learned GNN link score;
- neighborhood similarity;
- hierarchy depth;
- parent-child consistency;
- redundant ancestor avoidance.

---

## 13. Constraint Checking

Let:

\[
Q(f,r,c)\in\{0,1\}
\]

indicate whether a proposed relation satisfies graph constraints.

A proposal is rejected when it introduces:

- cycles;
- inverse inconsistency;
- invalid hierarchy;
- duplicate identity;
- self-links.

Only structurally admissible relations proceed.

---

## 14. Composite Relation Score

For candidate tag \(f\), relation \(r\), and concept \(c\):

\[
S_{\mathrm{rel}}(f,r,c)
=
\alpha S_{\mathrm{text}}
+
\beta S_{\mathrm{GNN}}
+
\gamma S_{\mathrm{beh}}
+
\eta S_{\mathrm{lex}}
\]

subject to:

\[
Q(f,r,c)=1.
\]

The best placement is:

\[
(c^\ast,r^\ast)
=
\arg\max_{c,r}
S_{\mathrm{rel}}(f,r,c).
\]

The weights may ultimately be learned from validated relations rather than fixed manually.

---

## 15. Existing Concept vs. New Concept

Let:

\[
E(f,c)
\]

measure equivalence evidence.

Define:

\[
E^\ast(f)
=
\max_{c\in\mathcal C_t}E(f,c).
\]

If:

\[
E^\ast(f)\ge\tau_{\mathrm{eq}},
\]

the new string may map to an existing concept or become an alternate label.

Otherwise it may persist as a folksonomic tag.

Thus:

\[
f
\rightarrow
\begin{cases}
\text{existing concept}\\
\text{alternate label}\\
\text{folksonomic tag}\\
\text{new controlled concept}\\
\text{reject}
\end{cases}
\]

---

## 16. Novel Concept Admission

A novel tag can be evaluated using several dimensions.

### Novelty

\[
N(f)
=
1-
\max_{c\in\mathcal C_t}
\cos(\mathbf x_f,\mathbf x_c).
\]

### Support

For \(n_f\) distinct uses and \(m_f\) distinct users:

\[
F(f)
=
\lambda_1\log(1+n_f)
+
\lambda_2\log(1+m_f).
\]

### Contextual coherence

For documents:

\[
D_f
=
\{d:Y_{df}=1\},
\]

define:

\[
C(f)
=
\frac{2}
{|D_f|(|D_f|-1)}
\sum_{i<j}
\cos(\mathbf x_{d_i},\mathbf x_{d_j}).
\]

### Placement confidence

\[
R(f)
=
\max_{c,r}
S_{\mathrm{rel}}(f,r,c).
\]

A general admission function is:

\[
A(f)
=
w_NN(f)
+
w_FF(f)
+
w_CC(f)
+
w_RR(f).
\]

Promotion occurs when:

\[
A(f)\ge\tau_A.
\]

The admission model can be calibrated empirically.

---

## 17. Confidence-Based Automation

Let \(p_1\) be the highest relation probability and \(p_2\) the second highest.

Automatic placement is permitted when:

\[
p_1\ge\tau_H
\]

and:

\[
p_1-p_2\ge\delta.
\]

Otherwise the system requests human validation.

---

## 18. Graph Update

For a validated new controlled concept \(f\):

\[
\mathcal C_{t+1}
=
\mathcal C_t\cup\{f\}.
\]

For accepted relation:

\[
(f,r^\ast,c^\ast),
\]

update:

\[
E_{t+1}
=
E_t
\cup
\{
(f,r^\ast,c^\ast)
\}.
\]

Therefore:

\[
\mathcal G_{t+1}
=
\mathcal G_t
+
\Delta\mathcal G_t.
\]

The central feedback loop is:

\[
\boxed{
\mathcal G_t
\rightarrow
\text{recommendations}
\rightarrow
\text{human tagging}
\rightarrow
\text{validated knowledge}
\rightarrow
\mathcal G_{t+1}
}
\]

---

## 19. Graph-Enhanced Recommendation

For document \(d\), let semantic seed concepts be:

\[
S_d
=
C_{\mathrm{sem}}^{(k)}(d).
\]

For graph candidate \(c\):

\[
s_{\mathrm{graph}}(c\mid d)
=
\sum_{s\in S_d}
w_s
\max_rP(r\mid s,c).
\]

Combine semantic and structural evidence:

\[
S_{\mathrm{tag}}(c\mid d)
=
\alpha s_{\mathrm{sem}}(d,c)
+
(1-\alpha)
s_{\mathrm{graph}}(c\mid d).
\]

Then:

\[
C_{\mathrm{graph}}(d)
=
\operatorname{TopK}
S_{\mathrm{tag}}(c\mid d).
\]

---

## 20. Relational Retrieval

For query \(q\), semantic retrieval produces:

\[
S_q
=
C_{\mathrm{sem}}^{(k)}(q).
\]

Graph expansion produces:

\[
G_q
=
\{
c:
s_{\mathrm{graph}}(c\mid S_q)>\tau_g
\}.
\]

The expanded concept representation is:

\[
Q^\ast
=
S_q\cup G_q.
\]

If \(T_d\) denotes document tags, define compatibility:

\[
K(c,t)
=
\begin{cases}
1 & c=t\\
\kappa_h(c,t) & \text{hierarchically related}\\
\kappa_a(c,t) & \text{associatively related}\\
0 & \text{otherwise}
\end{cases}
\]

and retrieval score:

\[
S_{\mathrm{ret}}(d\mid q)
=
\lambda
S_{\mathrm{text}}(d,q)
+
(1-\lambda)
\sum_{c\in Q^\ast}
w_c
\max_{t\in T_d}
K(c,t).
\]

This combines textual and relational retrieval.

---

## 21. Hierarchical Retrieval

For concept \(c\):

\[
\operatorname{Desc}(c)
=
\{
x:x\prec^\ast c
\}
\]

and:

\[
\operatorname{Anc}(c)
=
\{
x:c\prec^\ast x
\}.
\]

Expansion can be weighted by graph distance:

\[
w(c,x)
=
e^{-\rho d_{\mathcal G}(c,x)}.
\]

Nearby hierarchical relatives therefore contribute more strongly than distant ones.

---

## 22. Folksonomic Retrieval

Folksonomic tags remain usable before formal promotion.

For query \(q\):

\[
S_{\mathrm{folk}}(d,q)
=
\max_{f\in T_d}
\cos(\mathbf x_q,\mathbf x_f).
\]

If a folksonomic tag maps to controlled concept \(c\), retrieval can cross the folksonomy–ontology boundary.

---

## 23. Experimental Ablations

The architecture naturally supports three primary system variants.

### Manual baseline

\[
M_0
=
\text{manual tagging}.
\]

### Semantic mediation

\[
M_1
=
M_0
+
C_{\mathrm{sem}}.
\]

### Structural-semantic TagSynapse

\[
M_2
=
M_1
+
C_{\mathrm{graph}}
+
\text{relational inference}.
\]

An adaptive vocabulary-evolution condition can be represented as:

\[
M_3
=
M_2
+
\text{concept admission/evolution}.
\]

This separates:

\[
M_0\rightarrow M_1
\]

as the contribution of semantic mediation,

\[
M_1\rightarrow M_2
\]

as the contribution of graph structure, and:

\[
M_2\rightarrow M_3
\]

as the contribution of adaptive vocabulary evolution.

---

## 24. Compact System Definition

TagSynapse can be summarized as:

\[
\boxed{
\mathcal G_{t+1}
=
\Phi
\left(
\mathcal G_t,
H[
\operatorname{Retrieve}
(
D,
f_\theta,
\mathcal G_t,
\operatorname{GNN}_\psi
)
],
\operatorname{Rel}_\phi,
\operatorname{Usage}_t
\right)
}
\]

where:

- \(f_\theta\) provides semantic representation;
- `Retrieve` produces semantic and graph-mediated candidates;
- \(H\) represents human tagging decisions;
- \(\operatorname{Rel}_\phi\) infers typed conceptual relations;
- \(\operatorname{Usage}_t\) captures accumulated collaborative evidence;
- \(\Phi\) validates and updates the knowledge graph.

Conceptually:

\[
\boxed{
\text{Users organize information through the graph, and their organization of information changes the graph.}
}
\]

