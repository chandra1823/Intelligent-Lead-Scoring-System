"""Schema mapper tests — arbitrary CRM columns onto the canonical schema."""

import unittest

import pandas as pd

from ml.canonical import CANONICAL_FIELDS, FEATURE_COLUMNS, normalize_key
from ml.mapping import (
    AUTO_ACCEPT,
    MappingProposal,
    apply_mapping,
    mapping_coverage,
    merge_confirmations,
    propose_mapping,
)


def hubspot_like() -> pd.DataFrame:
    """Column names in the style HubSpot actually exports."""
    return pd.DataFrame(
        {
            "hs_object_id": [f"{i}" for i in range(60)],
            "hs_analytics_num_visits": list(range(60)),
            "hs_analytics_num_page_views": [i * 2 for i in range(60)],
            "hs_analytics_source": ["ORGANIC_SEARCH", "PAID_SEARCH"] * 30,
            "jobtitle": ["Manager", "Student"] * 30,
            "country": ["India", "USA"] * 30,
            "is_converted": [i % 2 for i in range(60)],
        }
    )


class TestProposeMapping(unittest.TestCase):
    def test_maps_hubspot_style_columns(self) -> None:
        proposal = propose_mapping(hubspot_like())

        self.assertEqual(proposal.mapping.get("hs_analytics_num_visits"), "total_visits")
        self.assertEqual(proposal.mapping.get("hs_object_id"), "external_id")
        self.assertEqual(proposal.mapping.get("jobtitle"), "occupation")
        self.assertEqual(proposal.mapping.get("country"), "country")
        self.assertEqual(proposal.mapping.get("is_converted"), "converted")

    def test_one_field_is_claimed_only_once(self) -> None:
        frame = pd.DataFrame(
            {
                "country": ["India"] * 40,
                "country_region": ["India"] * 40,
                "visits": list(range(40)),
            }
        )

        proposal = propose_mapping(frame)
        targets = list(proposal.mapping.values())

        self.assertEqual(len(targets), len(set(targets)), "a canonical field was claimed twice")

    def test_a_conflicted_column_falls_back_to_its_next_best_field(self) -> None:
        """
        A column whose best field is taken must still get its second choice
        rather than being dropped.
        """
        frame = pd.DataFrame(
            {
                "country": ["India", "USA"] * 30,
                "amount": [100.0, 250.0] * 30,
            }
        )

        proposal = propose_mapping(frame)

        self.assertEqual(proposal.mapping.get("country"), "country")
        self.assertEqual(proposal.mapping.get("amount"), "deal_value")

    def test_unknown_columns_are_reported_not_guessed(self) -> None:
        frame = pd.DataFrame({"internal_widget_ref": ["x"] * 40, "visits": list(range(40))})

        proposal = propose_mapping(frame)

        self.assertNotIn("internal_widget_ref", proposal.mapping)
        self.assertIn("internal_widget_ref", proposal.unmapped_columns)

    def test_low_confidence_matches_are_flagged_for_review(self) -> None:
        proposal = propose_mapping(pd.DataFrame({"country region": ["India", "USA"] * 30}))

        item = next(p for p in proposal.proposals if p.source_column == "country region")
        if item.canonical_field is not None and item.confidence < AUTO_ACCEPT:
            self.assertTrue(item.needs_review)

    def test_confirmed_mappings_are_never_overridden(self) -> None:
        frame = hubspot_like()
        confirmed = {"jobtitle": "seniority"}

        proposal = propose_mapping(frame, already_mapped=confirmed)

        self.assertEqual(proposal.mapping["jobtitle"], "seniority")

    def test_result_is_independent_of_column_order(self) -> None:
        frame = hubspot_like()
        reversed_frame = frame[list(reversed(frame.columns))]

        self.assertEqual(propose_mapping(frame).mapping, propose_mapping(reversed_frame).mapping)


class TestApplyAndConfirm(unittest.TestCase):
    def test_apply_mapping_renames_and_drops(self) -> None:
        frame = pd.DataFrame({"visits": [1, 2], "junk": ["a", "b"]})

        result = apply_mapping(frame, {"visits": "total_visits"})

        self.assertEqual(list(result.columns), ["total_visits"])

    def test_merge_confirmations_applies_corrections(self) -> None:
        proposal = MappingProposal(
            mapping={"a": "total_visits"}, proposals=[], unmapped_columns=[],
            missing_features=[], confident_count=0, review_count=0,
        )

        merged = merge_confirmations(proposal, {"b": "city"})

        self.assertEqual(merged, {"a": "total_visits", "b": "city"})

    def test_merge_confirmations_rejects_with_none(self) -> None:
        proposal = MappingProposal(
            mapping={"a": "total_visits"}, proposals=[], unmapped_columns=[],
            missing_features=[], confident_count=0, review_count=0,
        )

        self.assertEqual(merge_confirmations(proposal, {"a": None}), {})

    def test_merge_confirmations_rejects_unknown_field(self) -> None:
        proposal = MappingProposal(
            mapping={}, proposals=[], unmapped_columns=[],
            missing_features=[], confident_count=0, review_count=0,
        )

        with self.assertRaises(ValueError):
            merge_confirmations(proposal, {"a": "not_a_real_field"})

    def test_coverage_reports_target_presence(self) -> None:
        coverage = mapping_coverage({"a": "total_visits", "b": "converted"})

        self.assertTrue(coverage["has_target"])
        self.assertIn("total_visits", coverage["mapped_features"])
        self.assertGreater(len(coverage["missing_features"]), 0)


class TestCanonicalSchema(unittest.TestCase):
    def test_aliases_normalise_without_collision(self) -> None:
        """Two fields must not claim the same alias."""
        seen: dict[str, str] = {}
        for spec in CANONICAL_FIELDS:
            for alias in spec.aliases:
                key = normalize_key(alias)
                if key in seen:
                    self.assertEqual(
                        seen[key], spec.name, f"alias '{alias}' is claimed by two fields"
                    )
                seen[key] = spec.name

    def test_feature_columns_are_all_real_fields(self) -> None:
        names = {spec.name for spec in CANONICAL_FIELDS}
        for column in FEATURE_COLUMNS:
            self.assertIn(column, names)


if __name__ == "__main__":
    unittest.main()
