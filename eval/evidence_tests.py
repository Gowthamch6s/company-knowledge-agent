from app.graph import graph


TEST_CASES = [
    # --------------------------------------------------------
    # ANSWERABLE QUESTIONS
    # --------------------------------------------------------
    {
        "question": "How much vacation do employees get?",
        "expected": True,
    },
    {
        "question": "How many paid sick days do employees receive?",
        "expected": True,
    },
    {
        "question": "How often are employees paid?",
        "expected": True,
    },
    {
        "question": "Can employees work remotely?",
        "expected": True,
    },
    {
        "question": "What security measures are required for company laptops?",
        "expected": True,
    },
    {
        "question": "What happens during the final week of December?",
        "expected": True,
    },
    {
        "question": "When do formal performance reviews happen?",
        "expected": True,
    },
    {
        "question": "What are the standard working hours?",
        "expected": True,
    },
    {
        "question": "How should employees report harassment?",
        "expected": True,
    },
    {
        "question": "Who owns software created during employment?",
        "expected": True,
    },

    # --------------------------------------------------------
    # UNANSWERABLE QUESTIONS
    # --------------------------------------------------------
    {
        "question": "What is the company's maternity leave policy?",
        "expected": False,
    },
    {
        "question": "Does the company provide dental insurance?",
        "expected": False,
    },
    {
        "question": "How much does the company match for a 401k?",
        "expected": False,
    },
    {
        "question": "What is the parental leave policy?",
        "expected": False,
    },
    {
        "question": "Does the company reimburse employee travel expenses?",
        "expected": False,
    },
    {
        "question": "Does the company offer tuition reimbursement?",
        "expected": False,
    },
        # --------------------------------------------------------
    # PARAPHRASED ANSWERABLE QUESTIONS
    # --------------------------------------------------------
    {
        "question": "How many PTO days do full-time workers earn each year?",
        "expected": True,
    },
    {
        "question": "When does Nexus pay its employees?",
        "expected": True,
    },
    {
        "question": "Am I allowed to work from home?",
        "expected": True,
    },
    {
        "question": "What should I do if I witness workplace harassment?",
        "expected": True,
    },
    {
        "question": "When are employee performance evaluations conducted?",
        "expected": True,
    },
    {
        "question": "What protections are required on company-issued laptops?",
        "expected": True,
    },
    {
        "question": "What are Nexus's normal business hours?",
        "expected": True,
    },
    {
        "question": "How many sick days are available each year?",
        "expected": True,
    },
    {
        "question": "Does Nexus shut down at the end of December?",
        "expected": True,
    },
    {
        "question": "Does code I create while employed belong to me or the company?",
        "expected": True,
    },

    # --------------------------------------------------------
    # HARD UNANSWERABLE QUESTIONS
    # Related to real handbook topics, but NOT actually answered
    # --------------------------------------------------------
    {
        "question": "Can unused PTO be carried over to the next year?",
        "expected": False,
    },
    {
        "question": "Can employees cash out unused PTO?",
        "expected": False,
    },
    {
        "question": "How many weeks of paid parental leave are provided?",
        "expected": False,
    },
    {
        "question": "Can employees work remotely from another country?",
        "expected": False,
    },
    {
        "question": "Does Nexus pay for home office equipment?",
        "expected": False,
    },
    {
        "question": "Does the company reimburse employees for internet service?",
        "expected": False,
    },
    {
        "question": "How much overtime pay do employees receive?",
        "expected": False,
    },
    {
        "question": "Are employees paid annual bonuses?",
        "expected": False,
    },
    {
        "question": "What health insurance plans does Nexus offer?",
        "expected": False,
    },
    {
        "question": "What percentage does Nexus contribute to retirement accounts?",
        "expected": False,
    },
    {
        "question": "Can employees take more than 10 sick days if they are hospitalized?",
        "expected": False,
    },
    {
        "question": "Can employees use personal laptops for company work?",
        "expected": False,
    },
    {
        "question": "How often must employees change their passwords?",
        "expected": False,
    },
    {
        "question": "Does the December shutdown count against employee PTO?",
        "expected": False,
    },
]


def run_evaluation():
    correct = 0

    false_positives = 0
    false_negatives = 0

    print("\nEVIDENCE CHECKING EVALUATION")
    print("=" * 100)

    for number, test in enumerate(TEST_CASES, start=1):

        result = graph.invoke(
            {
                "question": test["question"],
                "evidence": [],
                "evidence_sufficient": False,
                "answer": "",
            }
        )

        predicted = result["evidence_sufficient"]
        expected = test["expected"]

        passed = predicted == expected

        if passed:
            correct += 1

        if predicted is True and expected is False:
            false_positives += 1

        if predicted is False and expected is True:
            false_negatives += 1

        status = "PASS" if passed else "FAIL"

        print(
            f"{number:>2}. "
            f"{status:<5} | "
            f"Expected: {str(expected):<5} | "
            f"Predicted: {str(predicted):<5} | "
            f"{test['question']}"
        )

    total = len(TEST_CASES)

    accuracy = correct / total

    print("\n" + "=" * 100)

    print(
        f"Accuracy: "
        f"{correct}/{total} = {accuracy:.1%}"
    )

    print(
        f"False positives: {false_positives}"
    )

    print(
        f"False negatives: {false_negatives}"
    )

    print("=" * 100)


if __name__ == "__main__":
    run_evaluation()