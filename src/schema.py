FEATURES = [
    ("Marital status", "marital_status", "nominal", "demographic"),
    ("Application mode", "application_mode", "nominal", "enrollment"),
    ("Application order", "application_order", "numeric", "enrollment"),
    ("Course", "course", "nominal", "enrollment"),
    ("Daytime/evening attendance", "daytime_evening_attendance", "binary", "enrollment"),
    ("Previous qualification", "previous_qualification", "nominal", "enrollment"),
    ("Previous qualification (grade)", "previous_qualification_grade", "numeric", "enrollment"),
    ("Nacionality", "nationality", "nominal", "demographic"),
    ("Mother's qualification", "mothers_qualification", "nominal", "socioeconomic"),
    ("Father's qualification", "fathers_qualification", "nominal", "socioeconomic"),
    ("Mother's occupation", "mothers_occupation", "nominal", "socioeconomic"),
    ("Father's occupation", "fathers_occupation", "nominal", "socioeconomic"),
    ("Admission grade", "admission_grade", "numeric", "enrollment"),
    ("Displaced", "displaced", "binary", "demographic"),
    ("Educational special needs", "educational_special_needs", "binary", "demographic"),
    ("Debtor", "debtor", "binary", "socioeconomic"),
    ("Tuition fees up to date", "tuition_fees_up_to_date", "binary", "socioeconomic"),
    ("Gender", "gender", "binary", "demographic"),
    ("Scholarship holder", "scholarship_holder", "binary", "socioeconomic"),
    ("Age at enrollment", "age_at_enrollment", "numeric", "demographic"),
    ("International", "international", "binary", "demographic"),
    ("Curricular units 1st sem (credited)", "cu_1st_credited", "numeric", "sem1"),
    ("Curricular units 1st sem (enrolled)", "cu_1st_enrolled", "numeric", "sem1"),
    ("Curricular units 1st sem (evaluations)", "cu_1st_evaluations", "numeric", "sem1"),
    ("Curricular units 1st sem (approved)", "cu_1st_approved", "numeric", "sem1"),
    ("Curricular units 1st sem (grade)", "cu_1st_grade", "numeric", "sem1"),
    ("Curricular units 1st sem (without evaluations)", "cu_1st_without_evaluations", "numeric", "sem1"),
    ("Curricular units 2nd sem (credited)", "cu_2nd_credited", "numeric", "sem2"),
    ("Curricular units 2nd sem (enrolled)", "cu_2nd_enrolled", "numeric", "sem2"),
    ("Curricular units 2nd sem (evaluations)", "cu_2nd_evaluations", "numeric", "sem2"),
    ("Curricular units 2nd sem (approved)", "cu_2nd_approved", "numeric", "sem2"),
    ("Curricular units 2nd sem (grade)", "cu_2nd_grade", "numeric", "sem2"),
    ("Curricular units 2nd sem (without evaluations)", "cu_2nd_without_evaluations", "numeric", "sem2"),
    ("Unemployment rate", "unemployment_rate", "numeric", "macro"),
    ("Inflation rate", "inflation_rate", "numeric", "macro"),
    ("GDP", "gdp", "numeric", "macro"),
]

TARGET = "Target"


def get_feature_columns(feature_set="B_full"):
    if feature_set not in {"A_early", "B_full"}:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return [csv_name for csv_name, _, _, group in FEATURES
            if feature_set == "B_full" or group != "sem2"]


def feature_by_id(feature_id):
    for csv_name, item_id, kind, group in FEATURES:
        if item_id == feature_id:
            return {"csv_name": csv_name, "id": item_id, "type": kind, "group": group}
    raise KeyError(feature_id)
