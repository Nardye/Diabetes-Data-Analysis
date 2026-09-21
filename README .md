# **Diabetes Hospital Readmission Analysis**

## **Project Overview**

Hospital readmissions are an important challenge for healthcare organizations because they can indicate complex patient needs, gaps in care coordination, and opportunities for improved post-discharge support.

This project analyzes **10,000 hospital encounters involving patients with diabetes** to identify patient, hospitalization, healthcare-utilization, diagnosis, and medication characteristics associated with hospital readmission.

The project follows the **CRISP-DM framework** and demonstrates an end-to-end data analytics workflow using **Python, Pandas, SQL, data visualization, and Power BI**.

> **Note:** This is a descriptive analytics project. The results identify patterns and associations and should not be interpreted as causal relationships.

---

## **🎯 Business Question**

**What patient, hospitalization, utilization, diagnosis, and medication characteristics are associated with hospital readmission?**

The analysis explores questions such as:

* What is the overall hospital readmission rate?  
* Which age groups have higher readmission rates?  
* Does readmission vary by gender or race?  
* How does previous inpatient or emergency utilization relate to readmission?  
* Do readmitted patients have longer hospital stays?  
* Which diagnoses combine high readmission rates with meaningful patient volume?  
* How do diabetes medication status and medication changes relate to readmission?  
* Which patient segments may warrant additional attention from hospital leadership?

---

## **🛠️ Tools & Technologies**

* **Python**  
* **Pandas**  
* **Matplotlib**  
* **SQL / SQLite**  
* **Jupyter Notebook**  
* **Power BI**  
* **CRISP-DM**

---

## **📊 Key Performance Indicators**

The analysis produced the following overall KPIs:

| KPI | Result |
| ----- | ----- |
| Total Encounters | 10,000 |
| Total Readmissions | 3,965 |
| Overall Readmission Rate | **39.65%** |
| Average Length of Stay | 4.43 days |
| Average Medications | 15.56 |
| Average Lab Procedures | 43.08 |
| Average Prior Inpatient Visits | 0.39 |
| Average Emergency Visits | 0.12 |
| Receiving Diabetes Medication | 74.78% |
| Medication Changes | 42.76% |

---

## **🔍 Key Findings**

### **1\. Overall Readmission**

Out of 10,000 hospital encounters, **3,965 resulted in readmission**, representing an overall readmission rate of **39.65%**.

### **2\. Age and Readmission**

Readmission generally increased among older age groups.

* Ages **70–80:** 43.04% readmission rate  
* Ages **80–90:** 44.32% readmission rate

The 70–80 age group was particularly important because it combined an elevated readmission rate with a substantial volume of **2,595 encounters**.

### **3\. Previous Healthcare Utilization**

Previous inpatient and emergency utilization showed some of the clearest relationships with readmission.

For previous inpatient visits:

* No previous inpatient visits: **34.36%**  
* One previous inpatient visit: **51.17%**  
* Two previous inpatient visits: **62.74%**

Previous emergency utilization showed a similar pattern. However, very high readmission percentages among groups with only a few encounters should be interpreted cautiously.

### **4\. Diagnosis and Readmission**

A high readmission percentage alone does not necessarily indicate the greatest operational importance. Both **readmission rate and encounter volume** should be considered.

Congestive heart failure was particularly notable:

* **636 encounters**  
* **342 readmissions**  
* **53.77% readmission rate**

This combination of relatively high readmission and meaningful patient volume makes the diagnosis operationally important.

### **5\. Length of Stay**

Readmitted patients had a slightly longer average hospital stay:

* Readmitted: **4.63 days**  
* Not readmitted: **4.31 days**

Patients whose length of stay exceeded the average for their age group also had a higher readmission rate:

* Above age-group average LOS: **41.89%**  
* At or below age-group average LOS: **38.09%**

### **6\. Diabetes Medication**

Patients receiving diabetes medication had a higher observed readmission rate:

* Receiving diabetes medication: **41.16%**  
* Not receiving diabetes medication: **35.17%**

Patients with medication changes also had a higher observed rate:

* Medication change: **42.54%**  
* No medication change: **37.49%**

These results show associations only. They should **not** be interpreted as evidence that diabetes medication or medication changes cause readmission.

---

## **🧹 Data Preparation**

The project includes several data-preparation and feature-engineering steps, including:

* Identifying missing values  
* Replacing special `?` indicators with missing values  
* Reviewing duplicate records  
* Cleaning categorical variables  
* Creating analytical features  
* Preparing data for Pandas and SQL analysis  
* Exporting a cleaned dataset for additional analysis and visualization

Special attention was given to data quality before interpreting results.

---

## **🐼 Pandas Analysis**

Advanced Pandas techniques were used to investigate:

* Readmission by demographic characteristics  
* Admission and discharge characteristics  
* Length of stay  
* Previous inpatient visits  
* Previous emergency visits  
* Previous outpatient visits  
* Overall healthcare utilization  
* Diagnosis-level readmission  
* Medication status  
* Medication changes  
* Multi-variable patient segments

The project also demonstrates:

* `groupby()`  
* `agg()`  
* `transform()`  
* `.loc[]`  
* `pivot_table()`  
* `value_counts()`

---

## **🗄️ SQL Analysis**

The cleaned data was analyzed using **SQLite**, with 14 analytical SQL queries covering:

* Overall readmission KPIs  
* Gender analysis  
* Age analysis  
* Race analysis  
* Admission type  
* Length of stay  
* Prior inpatient utilization  
* Diagnosis analysis  
* Diabetes medication  
* Medication changes

More advanced SQL techniques include:

* `GROUP BY`  
* `CASE WHEN`  
* Conditional aggregation  
* `HAVING`  
* Common Table Expressions (**CTEs**)  
* Window functions  
* Ranking

---

## **✅ Pandas vs. SQL Validation**

To verify the reliability of the analysis, key results calculated with Pandas were independently reproduced using SQL.

**All five KPIs included in the validation matched between Pandas and SQL.**

This cross-validation provides an additional quality check on the analytical workflow.

---

## **📈 Data Visualization**

Python visualizations were created to communicate major patterns in hospital readmission, including differences across patient characteristics, healthcare utilization, diagnoses, and hospitalization characteristics.

The project also supports development of an interactive **Power BI dashboard** for communicating results to hospital stakeholders.

---

## **💡 Business Recommendations**

Based on the descriptive findings, hospital leadership could give additional attention to groups that combine **higher readmission rates with meaningful encounter volume**, including:

* Older patients  
* Patients with previous inpatient utilization  
* Patients with previous emergency utilization  
* High-volume diagnoses such as congestive heart failure  
* Patients with longer hospital stays or more complex care needs

Potential areas for further evaluation include stronger discharge planning, follow-up care, medication review, care coordination, and early post-discharge support.

---

## **⚠️ Limitations**

Several limitations should be considered when interpreting the analysis.

`A1Cresult` and `max_glu_serum` contain substantial missing data, limiting conclusions involving glucose-control measurements.

Some high readmission rates also occur in categories with very few encounters. Therefore, **rate and sample size should always be considered together**.

The dataset does not contain several potentially important factors, including:

* Detailed disease severity  
* Medication adherence  
* Socioeconomic status  
* Follow-up care  
* Detailed comorbidity information  
* Reasons for readmission

Most importantly, the analysis identifies **associations rather than causal effects**.

---

## **🚀 Future Work**

Future versions of the project could extend the analysis by:

* Building a predictive model for readmission  
* Comparing logistic regression with machine-learning models  
* Evaluating model performance using precision, recall, F1-score, and ROC-AUC  
* Investigating feature importance  
* Adding more detailed comorbidity information  
* Developing an interactive Power BI dashboard  
* Examining readmission patterns using additional clinical and socioeconomic variables

---

## **📂 Repository Structure**

diabetes-readmission-analysis/  
│  
├── README.md  
├── diabetes\_readmission\_analysis.ipynb  
├── data/  
│   └── diabetes\_clean.csv  
├── requirements.txt  
└── .gitignore  
---

## **▶️ Running the Project**

Clone the repository:

git clone \<repository-url\>

Navigate to the project folder:

cd diabetes-readmission-analysis

Install the required Python packages:

pip install \-r requirements.txt

Then open the notebook:

jupyter notebook diabetes\_readmission\_analysis.ipynb  
---

## **👩‍💻 Author**

**Nardos**

First Data Science Portfolio Project

## **Disclaimer**

This project was created for educational and portfolio purposes. The findings should not be interpreted as medical advice or as causal evidence regarding hospital readmission.

