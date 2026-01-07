PAR EL YESRI INASS-24010318
<img width="100" height="150" alt="image" src="https://github.com/user-attachments/assets/8ff73355-eaf0-42d3-ac75-12fdc08df8d2" />


1\. Le Contexte Métier et la Mission
====================================

1.1 Le Problème (Business Case)
-------------------------------

Le dataset que j'ai utilisé décrit des **dossiers clients d’une institution financière** (type banque / organisme de crédit).Chaque ligne représente un **client** et regroupe des informations socio-économiques :

*   Sexe (CODE\_GENDER)
    
*   Possession d’une voiture (FLAG\_OWN\_CAR)
    
*   Possession d’un bien immobilier (FLAG\_OWN\_REALTY)
    
*   Nombre d’enfants (CNT\_CHILDREN)
    
*   Revenu annuel (AMT\_INCOME\_TOTAL)
    
*   Type de revenu (NAME\_INCOME\_TYPE)
    
*   Niveau d’éducation (NAME\_EDUCATION\_TYPE)
    
*   Statut familial (NAME\_FAMILY\_STATUS)
    
*   Type de logement (NAME\_HOUSING\_TYPE)
    
*   Âge (via DAYS\_BIRTH, nombre de jours avant la date actuelle)
    
*   Ancienneté professionnelle (DAYS\_EMPLOYED)
    
*   Indicateurs de contact (téléphone, email…)
    
*   Type de profession (OCCUPATION\_TYPE)
    
*   Nombre total de membres du foyer (CNT\_FAM\_MEMBERS)
    

Dans ton script, la **variable cible** utilisée pour la classification est la dernière colonne du CSV, c’est-à-dire :

> CNT\_FAM\_MEMBERS = nombre de personnes dans le foyer.

L’objectif du projet est donc, dans ce cadre pédagogique, de **construire un modèle de Machine Learning** capable de **prédire la taille du foyer** à partir de toutes les autres caractéristiques du client.

Même si ce n’est pas un cas « critique » comme le cancer dans le document de référence, ce type de modèle peut être utile pour :

*   **Segmentation marketing** : adapter les offres (assurances, prêts, cartes) aux familles nombreuses ou non.
    
*   **Analyse de risque** : comprendre si certaines configurations de foyer sont corrélées à des comportements de remboursement particuliers.
    
*   **Personnalisation produit** : proposer des produits adaptés aux célibataires, couples, familles nombreuses, etc.
    

1.2 Les Données (L’Input)
-------------------------

À partir de ton fichier Dataset.csv, on obtient :

*   **Nombre de lignes (clients)** : 438 557
    
*   **Nombre de colonnes (variables)** : 18
    
*   Variables **catégorielles** (texte ou indicateurs) et **numériques** (revenus, âge en jours, etc.) coexistent dans le même tableau.
    

Résumé rapide de quelques colonnes importantes :

*   AMT\_INCOME\_TOTAL : revenu annuel, très variable (jusqu’à plusieurs millions).
    
*   DAYS\_BIRTH : âge en jours (négatif, car exprimé « en jours avant aujourd’hui »).
    
*   DAYS\_EMPLOYED : ancienneté en emploi, avec certaines valeurs extrêmes (ex. 365243) qui peuvent représenter des codes particuliers (chômeur, inconnu…).
    
*   CNT\_FAM\_MEMBERS : prend au moins 13 valeurs distinctes (1, 2, 3, …, 20, etc.), ce qui en fait un **problème de classification multi-classes**.
    

2\. Le Code Python (Laboratoire)
================================

Cette partie décrit **ton script**, qui joue le rôle de « laboratoire » : on y charge les données, on les salit artificiellement, on les nettoie, on explore, on entraîne le modèle, puis on évalue les performances.

Ci-dessous, j’insère **ton code tel quel** (sans le modifier), afin qu’il soit clairement documenté dans le rapport.

```python
plt.figure(figsize=(6, 5))  # Ensure xticklabels and yticklabels match the actual number of classes in y_test/y_pred  # If y_test contains more classes than target_names assumes, this might error.  # Using unique classes from y_test/y_pred for labels if target_names is not suitable  plot_labels = report_target_names # Using the same labels as for the classification report  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,              xticklabels=plot_labels, yticklabels=plot_labels)  plt.xlabel('Prédiction')  plt.ylabel('Réalité')  plt.title('Matrice de Confusion')  plt.show()  print("\n--- FIN DU SCRIPT ---")
````

3\. Analyse Approfondie : Nettoyage (Data Wrangling)
====================================================

3.1 Simulation de données « sales »
-----------------------------------

Comme dans le document de référence, ton script commence par **simuler des données imparfaites**, pour se rapprocher de la réalité.

*   À partir des données propres df, tu crées une copie df\_dirty.
    
*   Tu introduis artificiellement des **valeurs manquantes** (NaN) dans **5 %** des cellules, sur **toutes les colonnes sauf target**.
    

L’idée est de reproduire le cas où :

*   Certains clients n’ont pas renseigné leur métier,
    
*   Le revenu, ou le nombre d’enfants, n’est pas toujours connu,
    
*   Des champs peuvent être manquants à cause de problèmes de saisie ou d’import de fichiers.
    

3.2 Séparation X / y
--------------------

Avant le nettoyage, tu sépares les données en :

*   **X** : toutes les colonnes explicatives (features) → df\_dirty.drop('target', axis=1)
    
*   **y** : la variable cible (ici, CNT\_FAM\_MEMBERS recopiée dans df\['target'\])
    

Cette séparation X/y est importante, car **y ne doit pas être modifiée** pendant le nettoyage : on ne veut pas imputer la cible.

3.3 Stratégie d’imputation
--------------------------

Tu utilises une **stratégie d’imputation différenciée** :

1.  numerical\_cols = X.select\_dtypes(include=np.number).columnsimputer\_numeric = SimpleImputer(strategy='mean')X\[numerical\_cols\] = imputer\_numeric.fit\_transform(X\[numerical\_cols\])
    
    *   Pour chaque colonne numérique, l’imputer calcule la **moyenne** sur toutes les lignes non manquantes.
        
    *   Les trous (NaN) sont remplacés par cette moyenne.
        
2.  categorical\_cols = X.select\_dtypes(exclude=np.number).columnsimputer\_categorical = SimpleImputer(strategy='most\_frequent')X\[categorical\_cols\] = imputer\_categorical.fit\_transform(X\[categorical\_cols\])
    
    *   Pour chaque colonne de type catégorie (genre, type de revenu, etc.), la valeur manquante est remplacée par la **modalité la plus fréquente** (le mode).
        

Tu crées ensuite X\_clean = X.copy(), qui contient la version **complètement imputée** des features.

3.4 Le Coin de l’Expert : Data Leakage
--------------------------------------

Comme dans le projet de référence, on peut noter une subtilité :

*   Tu fais l’imputation (fit de l’imputer) **sur l’ensemble des données** avant de couper en train/test.
    
*   En théorie, la **bonne pratique stricte** est :
    
    1.  Splitter en **train/test**,
        
    2.  **Fit** l’imputer sur le **train**,
        
    3.  **Transform** train **et** test avec ce même imputer.
        

Sinon, on parle de **data leakage** : les statistiques du test (moyenne, mode) « fuient » dans la phase d’entraînement.

Ici, comme il s’agit d’un projet pédagogique, cette approximation est acceptable, mais il est utile de connaître la version « production ready » (via un Pipeline Scikit-Learn par exemple).

4\. Analyse Approfondie : Exploration (EDA)
===========================================

Dans la partie EDA, tu cherches à **profilier** les clients et à comprendre la structure de tes données avant d’entraîner le modèle.

4.1 Statistiques descriptives
-----------------------------

Tu affiches des statistiques descriptives sur les premières colonnes numériques de X\_clean :

*   count, mean, std, min, 25%, 50%, 75%, max.
    

Ces indicateurs permettent de repérer :

*   **Variables très dispersées** (grand écart-type) comme le revenu AMT\_INCOME\_TOTAL.
    
*   **Variables très concentrées** (petit écart-type) qui apportent peu d’information (presque constantes).
    
*   **Valeurs extrêmes** ou aberrantes (âge en jours très grand ou supérieur à une vie humaine, par exemple).
    

4.2 Distribution des revenus par taille de foyer
------------------------------------------------

Tu choisis une variable clé pour l’analyse :

```python
feature_to_plot = 'AMT_INCOME_TOTAL'   `
```

Puis tu traces un **histogramme** :

*   Axe X : revenu total (AMT\_INCOME\_TOTAL)
    
*   Couleur (hue) : la classe cible (target = taille du foyer)
    

Ce graphique permet de voir :

*   Si les foyers nombreux ont en moyenne un revenu différent des foyers plus petits,
    
*   S’il existe des **segments de clients** (faible revenu / revenu moyen / très haut revenu) associés à des tailles de foyers spécifiques.
    

4.3 Corrélations entre variables numériques
-------------------------------------------

Tu calcules ensuite une **matrice de corrélation** sur les colonnes numériques de X\_clean et tu l’affiches avec :

```python
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")   `
```

Ce type de visualisation met en évidence :

*   **Redondance** entre variables (par exemple, un lien possible entre CNT\_CHILDREN et CNT\_FAM\_MEMBERS, même si cette dernière est la cible).
    
*   Relations entre âge (DAYS\_BIRTH), ancienneté (DAYS\_EMPLOYED) et revenu (AMT\_INCOME\_TOTAL).
    
*   Variables quasi indépendantes du reste.
    

Pour un Random Forest, la multicollinéarité n’est pas un problème majeur, mais pour d’autres modèles (régression logistique, SVM linéaire) elle peut rendre l’interprétation plus compliquée.

5\. Analyse Approfondie : Méthodologie (Split)
==============================================

Tu appliques ensuite le découpage train/test :

```python
X_train, X_test, y_train, y_test = train_test_split(      X_clean, y, test_size=0.2, random_state=42  )
```python 

*   **80 %** des données pour l’entraînement (train),
    
*   **20 %** pour le test (test),
    
*   random\_state=42 garantit la **reproductibilité** du split.
    

**Pourquoi c’est important ?**

*   Le modèle doit être évalué sur des données **jamais vues** pendant l’apprentissage pour estimer sa capacité à **généraliser**.
    
*   Le fait de fixer la graine (42) assure que toi et n’importe qui qui relance ton script obtiennent **exactement le même découpage**, donc les mêmes résultats.
    

6\. FOCUS THÉORIQUE : L’Algorithme Random Forest 🌲
===================================================

Tu utilises :

```python
  model = RandomForestClassifier(n_estimators=100, random_state=42)  model.fit(X_train, y_train)   `
```

Random Forest est un **ensemble d’arbres de décision** entraînés sur des sous-échantillons des données.

6.1 L’arbre de décision (l’individu)
------------------------------------

Un arbre de décision pose des **questions successives** :

*   Exemple :
    
    1.  AMT\_INCOME\_TOTAL > seuil1 ?
        
    2.  DAYS\_BIRTH < seuil2 ?
        
    3.  CNT\_CHILDREN > seuil3 ?etc.
        

Problème : un seul arbre est **très sensible au bruit**. Il peut :

*   Overfitter (apprendre par cœur des cas rares),
    
*   Produire des frontières de décision trop spécifiques.
    

6.2 Le bagging et la forêt
--------------------------

Random Forest corrige cela via deux sources d’aléa contrôlés :

1.  **Bootstrapping** des observations
    
    *   Chaque arbre voit un **échantillon différent** des clients (tirés avec remise).
        
    *   Les arbres n’apprennent pas tous la même « vision » du monde.
        
2.  **Sous-échantillonnage des variables (features)**
    
    *   À chaque split, l’arbre choisit la meilleure variable **parmi un sous-ensemble aléatoire** de colonnes.
        
    *   Cela force la forêt à utiliser des combinaisons variées de variables (revenu, âge, type de logement…), et pas seulement toujours la même.
        

6.3 Le vote majoritaire
-----------------------

Lors de la prédiction :

*   Chaque arbre propose une **classe** (dans ton cas, un nombre de membres du foyer).
    
*   Le Random Forest agrège ces prédictions par **vote majoritaire**.
    

Effet :

*   Les erreurs individuelles de certains arbres s’annulent,
    
*   Le signal global (les patterns vraiment robustes) ressort.
    

C’est pour cela qu’en pratique, Random Forest est un **excellent point de départ** pour des projets de classification tabulaire (comme ici).

7\. Analyse Approfondie : Évaluation (L’Heure de Vérité)
========================================================

Après l’entraînement, tu calcules :

```python

y_pred = model.predict(X_test)  acc = accuracy_score(y_test, y_pred)  print(f"   >>> Accuracy Score : {acc*100:.2f}%")  unique_test_labels = np.unique(y_test)  report_target_names = [str(int(label)) for label in unique_test_labels]  print(classification_report(y_test, y_pred,                              labels=unique_test_labels,                              target_names=report_target_names))  cm = confusion_matrix(y_test, y_pred)  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,              xticklabels=report_target_names,              yticklabels=report_target_names)   `
```

7.1 Accuracy globale
--------------------

L’accuracy mesure la **proportion de prédictions correctes** :

Accuracy=nombre de preˊdictions correctesnombre total de preˊdictions\\text{Accuracy} = \\frac{\\text{nombre de prédictions correctes}}{\\text{nombre total de prédictions}}Accuracy=nombre total de preˊdictionsnombre de preˊdictions correctes​

Dans un problème multi-classes comme ici (plusieurs tailles de foyers possibles), une bonne accuracy indique que l’algorithme arrive à capter une large partie de la structure du problème.

7.2 Rapport de classification
-----------------------------

Le classification\_report donne, pour chaque classe (1 membre, 2 membres, 3 membres, …) :

*   **Precision** : parmi les foyers prédits « taille 3 », quelle proportion est réellement de taille 3 ?
    
*   **Recall** : parmi tous les foyers réellement de taille 3, combien ont été correctement détectés ?
    
*   **F1-score** : moyenne harmonique de precision et recall, qui résume la performance de chaque classe.
    

Comme il y a **plusieurs classes**, on s’intéresse aussi aux moyennes (macro avg, weighted avg) qui donnent une vision globale de la qualité du modèle.

7.3 Matrice de confusion
------------------------

La matrice de confusion affiche, pour chaque **classe réelle**, la **répartition des prédictions** :

*   Ligne = valeur réelle (y\_test),
    
*   Colonne = valeur prédite (y\_pred),
    
*   Diagonale = prédictions correctes.
    

Elle permet de voir :

*   Si le modèle confond beaucoup **les foyers de 2 et 3 personnes**,
    
*   Si les classes rares (ex. 8, 9, 15 membres) sont mal prédites (classique en cas de **déséquilibre des classes**).
    

8\. Conclusion du Projet
========================

Ce projet montre comment appliquer un **pipeline complet de Data Science** sur un jeu de données tabulaires de type « clients bancaires » :

1.  **Contexte métier** : mieux comprendre et prédire des caractéristiques de la clientèle (ici, la taille du foyer) à partir de données socio-économiques.
    
2.  **Préparation des données** :
    
    *   Simulation de données manquantes,
        
    *   Imputation adaptée (moyenne pour les numériques, mode pour les catégorielles),
        
    *   Mise en garde sur le **data leakage**.
        
3.  **Exploration** :
    
    *   Statistiques descriptives,
        
    *   Visualisation de la distribution des revenus par classe cible,
        
    *   Corrélations entre variables.
        
4.  **Méthodologie expérimentale** :
    
    *   Découpage train/test reproductible,
        
    *   Entraînement d’un modèle robuste (Random Forest).
        
5.  **Évaluation** :
    
    *   Accuracy globale,
        
    *   Rapport de classification multi-classes,
        
    *   Matrice de confusion pour analyser plus finement les erreurs.
