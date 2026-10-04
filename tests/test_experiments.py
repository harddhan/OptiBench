from optibench.experiments import compare_optimizers, run_benchmark


def test_run_benchmark_and_compare():
    run = run_benchmark("sphere", "gradient_descent")
    assert run.function_name == "sphere"
    assert run.optimizer_name == "gradient_descent"
    assert run.success in (True, False)

    comparison = compare_optimizers("sphere")
    assert len(comparison) == 2
    assert {item.optimizer_name for item in comparison} == {"gradient_descent", "newton"}
