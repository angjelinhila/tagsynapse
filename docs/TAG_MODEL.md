# TagSynapse Formal Tag Model

This document specifies the representation, relation model, recommendation process, graph learning, graph evolution, and 
retrieval formalism underlying TagSynapse.

---

## 1. Objects, Users, and Tags

Let

\[
\mathcal D=\{d_1,\dots,d_N\}
\]

denote information objects, and

\[
\mathcal U=\{u_1,\dots,u_M\}
\]

denote users.

The initialized controlled vocabulary is

\[
\mathcal C_0=\{c_1,\dots,c_K\}.
\]

The user-generated folksonomic vocabulary at time \(t\) is

\[
\mathcal F_t=\{f_1,\dots,f_{J_t}\}.
\]

Importantly,

\[
\mathcal F_t\neq\mathcal C_t.
\]

A folksonomic tag may participate in the knowledge graph without becoming a controlled concept.

---

## 2. Tag Knowledge Graph

At time \(t\),

\[
\mathcal G_t=(V_t,E_t,\mathcal R)
\]

where

\[
V_t=\mathcal C_t\cup\mathcal F_t.
\]

Node provenance is represented by

\[
\tau(v)\in
\{
\text{controlled concept},
\text{folksonomic tag}
\}.
\]

Relations are partitioned as

\[
\mathcal R
=
\mathcal R_{\mathrm{ont}}
\cup
\mathcal R_{\mathrm{emp}}.
\]

Ontological relations include

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

Empirical relations include

\[
\mathcal R_{\mathrm{emp}}
=
\{
\texttt{coUsedWith},
\texttt{usedFor},
\texttt{createdBy}
\}.
\]

The two relation classes remain semantically distinct.

---

## 3. Hierarchy

Let

\[
c_i\prec c_j
\]

mean that \(c_i\) is narrower than \(c_j\).

The inverse relation is

\[
c_j\succ c_i.
\]

Hierarchy is transitive:

\[
c_i\prec c_j
\
