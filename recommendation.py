
def get_recommendations(
    attendance,
    study_hours,
    assignment_score,
    previous_score,
    predicted_score
):
    recommendations = []

    if attendance < 75:
        recommendations.append(
            "Improve Attendance and Attend Classes Regularly"
        )

    if study_hours < 2:
        recommendations.append(
            "Create a Daily Study Schedule and Increase Focused Study Time"
        )

    if previous_score < 50:
        recommendations.append(
            "Revise Previous Lessons and Solve Questions"
        )

    if assignment_score < 60:
        recommendations.append(
            "Revise Previous Lessons and Solve Practice Questions"
        )

    if predicted_score < 40:
        recommendations.append(
            "Follow a Structured Revision Plan and Seek Help from a Teacher"
        )
    elif predicted_score < 70:
        recommendations.append(
            "Practice More Questions and Take Weekly Mock Tests"
        )
    else:
        recommendations.append(
            "Maintain Consistent Study Habits and Practice Advanced Questions"
        )

    if not recommendations:
        recommendations.append(
            "Keep Following Your Study Routine and Monitor Your Progress"
        )

    return recommendations
