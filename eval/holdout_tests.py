HOLDOUT_TEST_CASES = [
    # =========================================================
    # ANSWERABLE — paraphrases not used in our main evaluation
    # =========================================================

    {
        "question": (
            "On which two dates each month does payroll "
            "normally run?"
        ),
        "expected": True,
    },
    {
        "question": (
            "What happens to payday when it falls on "
            "a federal holiday?"
        ),
        "expected": True,
    },
    {
        "question": (
            "How far in advance should I request fewer "
            "than three consecutive PTO days?"
        ),
        "expected": True,
    },
    {
        "question": (
            "How much notice is required for an extended "
            "leave longer than one week?"
        ),
        "expected": True,
    },
    {
        "question": (
            "Can sick leave be used to care for an "
            "immediate family member?"
        ),
        "expected": True,
    },
    {
        "question": (
            "What should I do if I unexpectedly become "
            "sick before my shift?"
        ),
        "expected": True,
    },
    {
        "question": (
            "Does the company normally close during the "
            "last week of the year?"
        ),
        "expected": True,
    },
    {
        "question": (
            "Which authentication protection is required "
            "on company laptops?"
        ),
        "expected": True,
    },
    {
        "question": (
            "Are employees required to use a VPN on "
            "company-issued laptops?"
        ),
        "expected": True,
    },
    {
        "question": (
            "May proprietary source code be entered into "
            "a public AI coding assistant?"
        ),
        "expected": True,
    },
    {
        "question": (
            "Where can an employee report unethical "
            "workplace behavior?"
        ),
        "expected": True,
    },
    {
        "question": (
            "Who owns system architecture developed "
            "during employment?"
        ),
        "expected": True,
    },

    # =========================================================
    # UNANSWERABLE — related topics with missing details
    # =========================================================

    {
        "question": (
            "What is the maximum amount of PTO an "
            "employee can accumulate?"
        ),
        "expected": False,
    },
    {
        "question": (
            "Are unused sick days carried into the "
            "following year?"
        ),
        "expected": False,
    },
    {
        "question": (
            "Can employees sell unused sick leave back "
            "to the company?"
        ),
        "expected": False,
    },
    {
        "question": (
            "Does Nexus provide employees with a "
            "company-paid cell phone?"
        ),
        "expected": False,
    },
    {
        "question": (
            "How many vacation days can employees borrow "
            "before they accrue them?"
        ),
        "expected": False,
    },
    {
        "question": (
            "Can employees permanently relocate overseas "
            "while remaining remote?"
        ),
        "expected": False,
    },
    {
        "question": (
            "How much money does Nexus contribute toward "
            "employee medical premiums?"
        ),
        "expected": False,
    },
    {
        "question": (
            "What is the severance package when an "
            "employee is terminated?"
        ),
        "expected": False,
    },
    {
        "question": (
            "How many months must someone work before "
            "becoming eligible for PTO?"
        ),
        "expected": False,
    },
    {
        "question": (
            "Are employees compensated for working "
            "during the December shutdown?"
        ),
        "expected": False,
    },
    {
        "question": (
            "Can employees disable MFA while working "
            "from home?"
        ),
        "expected": False,
    },
    {
        "question": (
            "How often does the company reimburse "
            "employees for home internet?"
        ),
        "expected": False,
    },
]