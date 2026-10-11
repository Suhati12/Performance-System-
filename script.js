// Preset sample datasets for live demonstration
const PRESETS = {
    high: {
        age: 21,
        gender: "Female",
        attendance: 94.5,
        study_hours: 8.5,
        assignment_score: 92.0,
        internal_marks: 46.0,
        previous_semester_marks: 88.0,
        extracurricular_hours: 4.0,
        internet_access: "Yes",
        sleep_hours: 7.5,
        backlogs: 0
    },
    avg: {
        age: 20,
        gender: "Male",
        attendance: 78.0,
        study_hours: 5.0,
        assignment_score: 72.0,
        internal_marks: 33.0,
        previous_semester_marks: 64.0,
        extracurricular_hours: 3.0,
        internet_access: "Yes",
        sleep_hours: 7.0,
        backlogs: 0
    },
    risk: {
        age: 22,
        gender: "Male",
        attendance: 52.0,
        study_hours: 2.0,
        assignment_score: 42.0,
        internal_marks: 17.0,
        previous_semester_marks: 42.0,
        extracurricular_hours: 1.0,
        internet_access: "No",
        sleep_hours: 5.5,
        backlogs: 3
    }
};

function loadPreset(presetKey) {
    const data = PRESETS[presetKey];
    if (!data) return;

    for (const [key, value] of Object.entries(data)) {
        const el = document.getElementById(key);
        if (el) {
            el.value = value;
            // Trigger change animation highlight
            el.classList.add('bg-warning', 'bg-opacity-25');
            setTimeout(() => {
                el.classList.remove('bg-warning', 'bg-opacity-25');
            }, 400);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Tooltips initialization
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
