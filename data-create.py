import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pandas as pd


class Student:
    max_grade = 20
    all_grades = np.array([])

    def __init__(self, true_grade, student_id, var, bias, avg):
        """Student has an id (for easier navigation in lists);
        a true grade, that is unaffected by graders neither bias nor grading variance;
        grading variance; bias, represented by discount depending on graders work;
        list of people graded by this student (IDs) and corresponding grades;
        list of people who graded this student (IDs)and corresponding grades;
        a median and an average of received grades"""

        self.id = student_id
        self.true_grade = true_grade
        self.var = var
        self.abs_bias = (avg - self.true_grade) * bias

        self.grades_given = np.array([])
        self.grades_received = np.array([])

        self.graders = np.array([])
        self.graded = np.array([])

        self.median_grade = None
        self.avg_grade = None

    @staticmethod
    def grade_restricted(given_grade, true_grade=-1):
        """Returns the grades, that satisfies common sense.
        Grade is an integer and cannot be less than a zero, or greater than the max_grade.
        Also satisfies an assumption that deviation from true grade is not greater than 3.
        The latter prevents situations where bias and variance too greatly discount (or increase) the grade."""
        given_grade = round(given_grade)
        true_grade = given_grade if true_grade == -1 else true_grade
        given_grade = min(given_grade, Student.max_grade, true_grade + 3)
        given_grade = max(given_grade, 0, true_grade - 3)
        return given_grade

    def grade_work(self, other):
        other_biased_grade = other.true_grade + self.abs_bias
        grade = np.random.normal(other_biased_grade, self.var)
        grade = self.grade_restricted(grade, other.true_grade)

        self.grades_given = np.append(self.grades_given, grade)
        self.graded = np.append(self.graded, other.id)

        other.grades_received = np.append(other.grades_received, grade)
        other.graders = np.append(other.graders, self.id)

        Student.all_grades = np.append(Student.all_grades, grade)


def create_students(size=1000, bias=0.0, true_mean=11, true_var=4):
    """Returns a list of Student instances. Students have random normal true grade, a variance of grading"""
    print('create')
    students = []
    for i in range(size):
        new_grade = np.random.normal(true_mean, true_var)
        new_grade = Student.grade_restricted(new_grade)
        var = abs(np.random.normal(0, 1))
        students.append(Student(new_grade, i, var, bias, true_mean))
    return students


def grading(students, size):
    print('grade')
    if size > len(students) - 1:
        raise ValueError
    else:
        for student in students:
            other_students = students.copy()
            other_students.remove(student)
            students_to_grade = np.random.choice(other_students, size)
            for other in students_to_grade:
                student.grade_work(other)
        for student in students:
            while len(student.grades_received) < size:
                other = np.random.choice(students)
                if other.id != student.id and other.id not in student.graders:
                    other.grade_work(student)
            student.median_grade = np.median(student.grades_received)
            student.avg_grade = np.mean(student.grades_received)


def calculate_deviations(students):
    print('calculating dev')
    rows = []
    for student in students:
        others_avg = np.array([])
        for i, other in enumerate(student.graded):
            received = list(students[int(other)].grades_received)
            received.remove(student.grades_given[i])
            others_avg = np.append(others_avg, np.mean(received))
        avg_deviation = np.mean(student.grades_given - others_avg)

        avg_given = np.mean(student.grades_given)
        all_others_grades = list(Student.all_grades)
        for grade in student.grades_received:
            all_others_grades.remove(grade)
        avg_all_others = np.mean(all_others_grades)

        row = (student.avg_grade, avg_deviation, avg_given - avg_all_others)
        rows.append(row)
    return np.array(rows)


def find_bias(data, estimation_type):
    print(f'calculating bias for {estimation_type}')
    model = LinearRegression()
    x = data.iloc[:, 0].to_numpy().reshape(-1, 1)
    y = data[estimation_type].to_numpy().reshape(-1, 1)
    model.fit(x, y)
    return -model.coef_[0][0]


Student.max_grade = 20
students_number = 300

students_bias = 0.07
true_grade_expected = 11
true_grade_variance = 2

students_list = create_students(size=students_number, bias=students_bias, true_mean=true_grade_expected,
                                true_var=true_grade_variance)
grading(students_list, 3)
students_data = pd.DataFrame(calculate_deviations(students_list), columns=["grade", "loc_dev", "all_dev"])

for i, estimation_type in enumerate(['all_dev', 'loc_dev']):
    bias_reg = find_bias(students_data, estimation_type)
    print(f"""
        Estimation is {bias_reg}, true value is {students_bias}.
        Error is {bias_reg - students_bias}, or {round(100 * (bias_reg - students_bias) / students_bias, 2)}%
        """)
    sns.regplot(x=students_data.iloc[:, 0], y=students_data[estimation_type])

plt.show()
