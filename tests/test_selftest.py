def test_selftest_passes(tmp_path, monkeypatch):
    """--selftest 是验证安装/打包是否完整的入口，本身也必须保持可用。"""
    monkeypatch.chdir(tmp_path)  # 结果文件写在当前目录
    from medminer.main import main
    assert main(["--selftest"]) == 0
    text = (tmp_path / "selftest_result.txt").read_text(encoding="utf-8")
    assert text.rstrip().endswith("SELFTEST OK") and "[FAIL]" not in text
