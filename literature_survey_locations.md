# Literature Survey: Exact Locations of Matrix Claims in the Papers

This guide maps every claim in your B.Tech Major Project Zeroth Review literature survey matrix directly to its **exact location** (Section name, Page number, Figure/Table reference, and Paragraph context) in the original papers. 

This serves as a cheat sheet for your presentation defense. If any evaluator asks: *"Where in the paper is this written?"*, you can immediately show them the exact page and section using the references below.

---

## 📄 1. Paper 1: Arshad et al. (2026)
**Title:** *Beyond glucose monitoring: Multi-analyte biosensing and AI-integrated wearable platforms for next-generation diabetes management.* (Sensors and Actuators Reports, 12, 100481)

* ### 🔍 Title & Journal
  * **Location:** Page 1, Top Left (Journal Name: *Sensors and Actuators Reports*), and Page 1, Lines 2--3 (Paper Title).
* ### 🔍 Methodology & Algorithms
  * **Claim:** *"Combined chemical and optical tracking tools with AI algorithms, virtual twin models, and automated feedback loops."*
  * **Exact Location:** 
    * **Page 8, Section 4 ("AI-integrated predictive sensing systems"), Lines 522--540:** Explains the integration of AI/ML with multi-analyte CGMs to build patient-specific digital twins that drive closed-loop insulin delivery.
    * **Page 8, Figure 8:** Diagram outlines the integrated pipeline showing how sensors stream data into AI/ML models (deep learning, neural networks) and output closed feedback loops.
    * **Page 2, Section 1 ("Introduction"), Lines 111--129:** Discusses combining multi-analyte continuous sensors with AI algorithms to create automated closed-loop diagnostic feedback loops.
* ### 🔍 Datasets & Inputs
  * **Claim:** *"Continuous bodily fluid data (sweat and tears) measuring a combination of insulin, lactate, and glucose levels."*
  * **Exact Location:**
    * **Page 3, Section 2 ("Working of multianalyte continuous monitoring systems"):** Discusses targeting interstitial fluid, blood, sweat, and tears.
    * **Page 3, Section 2.1 ("Electrochemical sensors with multi-analyte platforms"), Lines 190--200:** Documents sensors tracking glucose, lactate, and electrolytes.
    * **Page 5, Figure 4 (Caption) & Page 6, Lines 380--395:** Explains Tear-based monitoring of glucose and other analytes via smart contact lenses.
    * **Page 9, Table 2 ("Different wearable CGMs explored..."):** Lists specific devices using tears and sweat to measure glucose, lactate, and ketones.
* ### 🔍 Results
  * **Claim:** *"Monitoring several biological markers at the same time greatly improves forecasting accuracy and proactive treatment compared to standard glucose-only methods."*
  * **Exact Location:**
    * **Page 1, Abstract, Lines 27--40:** Explicitly states that single-analyte tracking provides limited metabolic insight and that multi-analyte biosensing platforms enable predictive modeling and proactive care.
    * **Page 11, Section 4 ("AI-integrated predictive sensing systems"), Lines 756--768:** Describes how the "digital twin" virtual model uses multi-analyte data (CGM, diet logs, wearables) to simulate responses, optimize dosages, and predict future glucose levels in real-time.
* ### 🔍 Limitations
  * **Claim:** *"Currently restricted to early lab models. Major challenges include scaling up manufacturing, lack of extensive human trials, and potential data privacy flaws in cloud storage."*
  * **Exact Location:**
    * **Page 1, Abstract, Lines 37--40:** Evaluates the translational readiness, noting the gap in scaled manufacturing and clinical trials.
    * **Page 14, Section 6.2 ("Low-cost assembly and scalable manufacturing"):** Outlines barriers in mass-production scaling.
    * **Page 10, Figure 8 (Caption):** Mentions the requirement for data privacy, security, and clinical validation trials.
    * **Page 11, Section 4.1 ("Privacy, Security and Autonomy in AI-integrated Biosensors"), Lines 786--802:** Specifically details data privacy, patient security, and cloud safety vulnerabilities.

---

## 📄 2. Paper 2: Olawade et al. (2026)
**Title:** *Digital twin paradigm in diabetes prediction and management.* (Diabetes Research and Clinical Practice, 231, 113075)

* ### 🔍 Title & Journal
  * **Location:** Page 1, Header (Journal Name: *Diabetes Research and Clinical Practice*), and Page 1, Lines 35--36 (Paper Title).
* ### 🔍 Methodology & Algorithms
  * **Claim:** *"Built individualized digital replicas using flexible AI systems that constantly refresh based on live patient data."*
  * **Exact Location:**
    * **Page 1, Abstract, Lines 43--52:** States that digital twin technology creates virtual replicas through computational modeling and real-time data integration.
    * **Page 2, Section 1 ("Introduction"), Figure 1 ("Conceptual framework..."):** Illustrates the integration of real-time data to update the virtual patient replica.
* ### 🔍 Datasets & Inputs
  * **Claim:** *"Merged inputs from CGMs, smartwatches, dietary records, and standard hospital lab tests."*
  * **Exact Location:**
    * **Page 2, Section 1 ("Introduction"), Figure 1:** Outlines inputs including CGMs, wearables (smartwatches), nutrition/dietary logs, and clinical/laboratory EHR databases.
    * **Page 5, Section 3.3, Line 514:** Lists inputs under "Demographics, biomarkers, lifestyle."
* ### 🔍 Results
  * **Claim:** *"Accurately projected glucose behaviors and future complications. Automated intervention loops increased the amount of time patients stayed in safe blood sugar zones by 6 to 11 percent."*
  * **Exact Location:**
    * **Page 4, Section 3.3 ("Complication risk prediction"), Lines 400--410:** Focuses on predicting microvascular (retinopathy, neuropathy) and macrovascular complications using digital twins.
    * **Page 6, Section 4.1 ("Closed-loop insulin delivery"), Lines 590--594:** Explicitly states: *"...reporting time in range increases of approximately 6 to 11 percentage points..."*
* ### 🔍 Limitations
  * **Claim:** *"Relies heavily on brief initial studies. Scaling this technology is difficult due to privacy worries, missing universal guidelines, and unequal AI accuracy across different demographics."*
  * **Exact Location:**
    * **Page 6, Lines 595--602:** Outlines study limitations, noting that studies are short-term (6--12 months) and mostly enroll high-income, tech-savvy users.
    * **Page 7, Section 5 ("Challenges and barriers to implementation"), Table 3:** Specifically details data privacy concerns, surveillance risks, algorithm bias, and unequal performance across demographic groups.

---

## 📄 3. Paper 3: Kiran et al. (2026)
**Title:** *A digital twin framework for predicting and simulating type 2 diabetes onset using retrospective lifestyle data.* (Frontiers in Digital Health, 8:1710829)

* ### 🔍 Title & Journal
  * **Location:** Page 1, CITATION block (Journal Name: *Frontiers in Digital Health*), and Page 1, Title Section.
* ### 🔍 Methodology & Algorithms
  * **Claim:** *"Used Cox survival models and causal graphs (DAGs) to test 'what-if' scenarios regarding lifestyle changes."*
  * **Exact Location:**
    * **Page 1, Abstract ("Methods" section), Lines 71--76:** Mentions employing a penalized Cox proportional hazards model and causal Directed Acyclic Graphs (DAGs) to explore counterfactual simulations.
    * **Page 4, Section 2 ("Survival Modeling Core") & Page 15, Section 9 ("Causal Inference & Counterfactual Simulations"):** Outlines the Cox mathematical core and DAG structure.
* ### 🔍 Datasets & Inputs
  * **Claim:** *"Historical data spanning 17 years from nearly 20,000 UK Biobank participants, including mental health, food intake, and daily activity metrics."*
  * **Exact Location:**
    * **Page 1, Abstract ("Methods" section), Lines 70--71:** States: *"...drawn from 19,774 participants in the UK Biobank cohort, followed for up to 17 years."*
    * **Page 8, Section 7 ("Data Preprocessing & Cohort Description"), Lines 680--695:** Confirms the sample size ($19,774$) and $17$-year follow-up period.
    * **Page 5, Section 7.1 ("Predictor Variables"):** Lists inputs including lifestyle (diet, activity), sleep, and mental health/loneliness indicators.
* ### 🔍 Results
  * **Claim:** *"Achieved high reliability in forecasting disease onset. The study confirmed that psychological stress and loneliness quicken diabetes development, particularly when paired with junk food diets."*
  * **Exact Location:**
    * **Page 1, Abstract ("Results" section), Lines 78--83:** States: *"...model demonstrated strong predictive performance (C-index = 0.90)... Psychosocial stressors such as loneliness, insomnia, and poor mental health were significant predictors..."*
    * **Page 22, Section 10 ("Results & Discussion"):** Confirms that stress and loneliness lead to poor dietary choices (processed foods), compounding risk.
* ### 🔍 Limitations
  * **Claim:** *"Dependent on participants’ memories of their lifestyle habits instead of objective sensor data. The causal framework might overlook unrecorded environmental or genetic variables."*
  * **Exact Location:**
    * **Page 27, Section 11 ("Limitations"), Lines 2230--2237:** States: *"...exposures and mediators... were derived from self-reported data. Such measures may be subject to recall bias..."*
    * **Page 27, Section 11, Lines 2225--2229:** Acknowledges the risk of residual or unmeasured confounding (missing environmental/genetic variables).

---

## 📄 4. Paper 4: Shamanna et al. (2024)
**Title:** *Personalized nutrition in type 2 diabetes remission: application of digital twin technology for predictive glycemic control.* (Frontiers in Endocrinology, 15:1485464)

* ### 🔍 Title & Journal
  * **Location:** Page 1, CITATION block (Journal Name: *Frontiers in Endocrinology*), and Page 1, Title Section.
* ### 🔍 Methodology & Algorithms
  * **Claim:** *"Blended machine learning techniques (CatBoost, LSTMs, Random Forests) to forecast glucose peaks and deliver immediate dietary recommendations."*
  * **Exact Location:**
    * **Page 6, Section 2.2 ("Model Selection and Machine Learning Algorithms"), Lines 425--433:** Specifically lists *CatBoostRegressor* and *Random Forest*.
    * **Page 9, Section 2.3 ("Prediction Core / Architecture"), Line 718:** Details the *Long Short-Term Memory (LSTM)* network for capturing temporal sequences.
* ### 🔍 Datasets & Inputs
  * **Claim:** *"Live CGM readings merged with step counters, sleep data, manual food entries, and historical clinical lab reports."*
  * **Exact Location:**
    * **Page 1, Abstract ("Objective" section) & Page 4, Section 2.1 ("Participants and Data Sources"):** Mentions combining real-time CGM data, steps, sleep, and manual meal logs.
* ### 🔍 Results
  * **Claim:** *"Lowered overall HbA1c scores substantially. The program allowed 94 percent of users to quit their prescriptions, with almost three-quarters sustaining disease remission."*
  * **Exact Location:**
    * **Page 12, Section 3 ("Results"), Lines 947--954:** States: *"...mean HbA1c improved remarkably from 9.0 to 6.1... After one year, 94% of the DT group discontinued all T2D medications, with 72.7% achieving T2D remission..."*
* ### 🔍 Limitations
  * **Claim:** *"The system’s success demands intense user dedication to tracking meals manually and wearing devices daily. Its effectiveness in lower-income populations remains unknown."*
  * **Exact Location:**
    * **Page 15, Section 4 ("Discussion"), Lines 1302--1313:** Details limitations including generalizability to varying socioeconomic conditions (lower-income) and challenges with app fatigue, decision fatigue, and meal-tracking adherence.

---

## 📄 5. Paper 5: Zhang et al. (2024)
**Title:** *A framework towards digital twins for type 2 diabetes.* (Frontiers in Digital Health, 6:1336050)

* ### 🔍 Title & Journal
  * **Location:** Page 1, CITATION block (Journal Name: *Frontiers in Digital Health*), and Page 1, Title Section.
* ### 🔍 Methodology & Algorithms
  * **Claim:** *"Applied regularized regression models and graph-based network algorithms to map connections within a vast medical knowledge database."*
  * **Exact Location:**
    * **Page 5, Section 2.2 ("Machine learning models"), Line 320:** Mentions using *LassoCV* (regularized regression) and Random Forest.
    * **Page 5, Section 2.3 ("Knowledge graph analyses"), Lines 297--315:** Documents using *Topic PageRank* and *Steiner Tree approximation* algorithms over the SPOKE graph database.
* ### 🔍 Datasets & Inputs
  * **Claim:** *"The Arivale dataset, featuring over a thousand molecular blood markers and standard clinical tests tracked over several months."*
  * **Exact Location:**
    * **Page 2, Section 2.1 ("Dataset description and processing"), Lines 174--195:** Details using the *Arivale dataset* containing $1,042$ multiomic blood markers (proteomics, metabolomics, clinical labs) tracked over $6$ and $12$ months.
* ### 🔍 Results
  * **Claim:** *"Advanced biological indicators (like proteins and metabolites) forecasted 6-to-12-month health shifts far better than basic age or weight metrics, discovering novel cellular targets."*
  * **Exact Location:**
    * **Page 1, Abstract ("Results and discussion" section) & Page 6, Section 3.1 ("Predicting clinical trajectories"):** Reports that multiomic indicators outperformed demographic/anthropometric variables (age, weight) for $6$--$12$ month forecasts, identifying molecular targets like $AKT1$ and $TYRO3$.
* ### 🔍 Limitations
  * **Claim:** *"Because the participants were primarily healthy, the findings do not easily apply to severe diabetes cases. The small group size limits confidence in long-term predictions."*
  * **Exact Location:**
    * **Page 8, Section 4 ("Discussion" - Limitations), Lines 538--552:** States: *"...sparsity of longitudinal data... limited sample size ($639$ subjects at 1 year)... data predominantly originated from generally healthy participants, building models for progression specifically within a T2D disease state was not feasible."*
