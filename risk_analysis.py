def evaluate_student_risk(student_data, pred_final_marks=None, pred_result=None):
    """
    Evaluates student risk level based on academic parameters and ML predictions.
    
    Parameters:
    student_data (dict): Dictionary with student parameters
    pred_final_marks (float, optional): ML predicted final marks
    pred_result (str, optional): ML predicted result ('PASS' or 'FAIL')
    
    Returns:
    dict: {
        'risk_level': 'High Risk' | 'Medium Risk' | 'Low Risk',
        'risk_score': float (0 - 100),
        'risk_factors': list of str,
        'recommendations': list of str
    }
    """
    attendance = float(student_data.get('attendance', 75))
    study_hours = float(student_data.get('study_hours', 5))
    internal_marks = float(student_data.get('internal_marks', 30))
    previous_marks = float(student_data.get('previous_semester_marks', 60))
    backlogs = int(student_data.get('backlogs', 0))
    assignment_score = float(student_data.get('assignment_score', 65))
    
    risk_factors = []
    recommendations = []
    risk_points = 0
    
    # 1. Attendance Check
    if attendance < 60.0:
        risk_points += 30
        risk_factors.append(f"Severely low attendance ({attendance:.1f}%, below 60% threshold)")
        recommendations.append("Immediate attendance recovery plan required. Contact class coordinator.")
    elif attendance < 75.0:
        risk_points += 15
        risk_factors.append(f"Moderate attendance deficiency ({attendance:.1f}%, below target 75%)")
        recommendations.append("Improve lecture attendance to meet 75% eligibility requirement.")

    # 2. Backlogs Check
    if backlogs >= 3:
        risk_points += 35
        risk_factors.append(f"Critical backlog count ({backlogs} active backlogs)")
        recommendations.append("Enrol in remedial classes for backlog subject clearing.")
    elif backlogs >= 1:
        risk_points += 15
        risk_factors.append(f"Active backlogs detected ({backlogs} subject backlog)")
        recommendations.append("Allocate dedicated weekend hours to backlog subject preparation.")

    # 3. Internal Marks Check (Max 50)
    if internal_marks < 20.0:
        risk_points += 25
        risk_factors.append(f"Low internal assessment score ({internal_marks:.1f}/50)")
        recommendations.append("Submit additional assignment re-evaluations and attend tutorial sessions.")
    elif internal_marks < 30.0:
        risk_points += 10
        risk_factors.append(f"Sub-optimal internal score ({internal_marks:.1f}/50)")

    # 4. Daily Study Hours
    if study_hours < 3.0:
        risk_points += 20
        risk_factors.append(f"Insufficient daily self-study time ({study_hours:.1f} hours/day)")
        recommendations.append("Increase structured daily self-study time to at least 4 hours.")
    elif study_hours < 4.5:
        risk_points += 10
        risk_factors.append(f"Below average study hours ({study_hours:.1f} hours/day)")

    # 5. Previous Semester Marks
    if previous_marks < 45.0:
        risk_points += 20
        risk_factors.append(f"Weak prior academic record ({previous_marks:.1f}% previous semester)")
        recommendations.append("Seek peer mentoring for core conceptual subjects.")

    # 6. ML Model Predictions Integration
    if pred_result == 'FAIL':
        risk_points += 25
        risk_factors.append("ML Classifier predicts HIGH PROBABILITY OF FAIL")
    
    if pred_final_marks is not None and pred_final_marks < 40.0:
        risk_points += 20
        risk_factors.append(f"ML Regressor projects failing final score ({pred_final_marks:.1f}/100)")

    # Normalize risk score between 0 and 100
    risk_score = min(100, risk_points)

    if risk_score >= 50 or backlogs >= 3 or (pred_result == 'FAIL' and attendance < 65):
        risk_level = 'High Risk'
    elif risk_score >= 20:
        risk_level = 'Medium Risk'
    else:
        risk_level = 'Low Risk'

    if not risk_factors:
        risk_factors.append("No critical risk factors identified. Consistent academic behavior observed.")
        recommendations.append("Maintain current study schedules and participation.")

    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'risk_factors': risk_factors,
        'recommendations': recommendations
    }
