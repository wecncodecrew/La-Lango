# tests/test_evaluation.py
#
# Unit tests for BLEU and chrF evaluation metrics.

from lalango.evaluation.bleu import get_ngrams, clipped_precision, bleu_score, corpus_bleu
from lalango.evaluation.chrf import chrf_sentence, corpus_chrf


class TestGetNgrams:

    def test_unigrams(self):
        tokens = ["a", "b", "c"]
        ngrams = get_ngrams(tokens, 1)
        assert ngrams[("a",)] == 1
        assert ngrams[("b",)] == 1
        assert ngrams[("c",)] == 1

    def test_bigrams(self):
        tokens = ["a", "b", "c"]
        ngrams = get_ngrams(tokens, 2)
        assert ngrams[("a", "b")] == 1
        assert ngrams[("b", "c")] == 1
        assert len(ngrams) == 2

    def test_repeated_tokens(self):
        tokens = ["a", "a", "a"]
        ngrams = get_ngrams(tokens, 1)
        assert ngrams[("a",)] == 3

    def test_sequence_shorter_than_n(self):
        tokens = ["a"]
        ngrams = get_ngrams(tokens, 2)
        assert len(ngrams) == 0


class TestClippedPrecision:

    def test_perfect_match(self):
        hyp = list("hello")
        ref = list("hello")
        precision = clipped_precision(hyp, ref, n=1)
        assert precision == 1.0

    def test_no_match(self):
        hyp = list("abc")
        ref = list("xyz")
        precision = clipped_precision(hyp, ref, n=1)
        assert precision == 0.0

    def test_clipping_prevents_gaming(self):
        # If hypothesis just repeats "a" 10 times, and reference has "a" once,
        # the clipped count should be 1, not 10.
        hyp = list("aaaaaaaaaa")   # 10 a's
        ref = list("a")            # 1 a
        precision = clipped_precision(hyp, ref, n=1)
        # matching = 1, total = 10 → 0.1
        assert abs(precision - 0.1) < 1e-6

    def test_partial_match(self):
        hyp = list("hello")
        ref = list("hell")
        # 4 out of 5 characters match at unigram level
        precision = clipped_precision(hyp, ref, n=1)
        assert abs(precision - 0.8) < 1e-6


class TestBLEUScore:

    def test_perfect_translation(self):
        score = bleu_score("hello world", "hello world")
        assert score == 100.0

    def test_empty_hypothesis(self):
        score = bleu_score("", "hello world")
        assert score == 0.0

    def test_no_overlap(self):
        score = bleu_score("xyz xyz xyz", "abc abc abc")
        assert score == 0.0

    def test_higher_is_better(self):
        """A more accurate translation should get a higher BLEU score."""
        reference = "How are you doing today"
        good_hyp = "How are you doing today"
        bad_hyp = "What is your name please"
        assert bleu_score(good_hyp, reference) > bleu_score(
            bad_hyp, reference
        )

    def test_brevity_penalty_applied(self):
        """A very short hypothesis should get a lower score than a full one."""
        reference = "How are you doing today"
        full_hyp = "How are you doing today"
        short_hyp = "How"
        assert bleu_score(full_hyp, reference) > bleu_score(short_hyp, reference)

    def test_score_between_0_and_100(self):
        for hyp, ref in [
            ("hello", "world"),
            ("abc def", "abc xyz"),
            ("test sentence here", "test sentence there"),
        ]:
            score = bleu_score(hyp, ref)
            assert 0.0 <= score <= 100.0


class TestCorpusBLEU:

    def test_perfect_corpus(self):
        hyps = ["hello", "world", "test"]
        refs = ["hello", "world", "test"]
        score = corpus_bleu(hyps, refs)
        assert score == 100.0

    def test_empty_corpus(self):
        score = corpus_bleu([], [])
        assert score == 0.0

    def test_corpus_score_stable(self):
        """Corpus BLEU should be consistent across multiple calls."""
        hyps = ["hello world", "how are you", "test sentence"]
        refs = ["hello world", "how are you doing", "test sentence here"]
        score1 = corpus_bleu(hyps, refs)
        score2 = corpus_bleu(hyps, refs)
        assert score1 == score2


class TestChrF:

    def test_perfect_match(self):
        score = chrf_sentence("hello", "hello")
        assert score == 100.0

    def test_empty_strings(self):
        assert chrf_sentence("", "hello") == 0.0
        assert chrf_sentence("hello", "") == 0.0

    def test_no_overlap(self):
        # "aaa" and "zzz" share no character n-grams
        score = chrf_sentence("aaa", "zzz")
        assert score == 0.0

    def test_higher_is_better(self):
        reference = "How are you doing today"
        good_hyp = "How are you doing today"
        bad_hyp = "Something completely different here"
        assert chrf_sentence(good_hyp, reference) > chrf_sentence(
            bad_hyp, reference
        )

    def test_score_between_0_and_100(self):
        for hyp, ref in [
            ("hello", "world"),
            ("abc", "abx"),
        ]:
            score = chrf_sentence(hyp, ref)
            assert 0.0 <= score <= 100.0

    def test_corpus_chrf(self):
        hyps = ["hello", "world"]
        refs = ["hello", "world"]
        score = corpus_chrf(hyps, refs)
        assert score == 100.0
