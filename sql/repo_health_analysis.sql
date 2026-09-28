-- OSS Health Intelligence: SQL analysis (SQLite)
-- Table: repos (loaded from data/clean_labeled_repos.csv)

-- Q1: Does popularity (star tier) predict risk?
SELECT
    star_tier,
    COUNT(*) AS total_repos,
    SUM(CASE WHEN risk_category = 'High' THEN 1 ELSE 0 END) AS high_risk_count,
    ROUND(100.0 * SUM(CASE WHEN risk_category = 'High' THEN 1 ELSE 0 END) / COUNT(*), 1) AS high_risk_pct
FROM repos
GROUP BY star_tier
ORDER BY CASE star_tier
    WHEN 'mega' THEN 1 WHEN 'large' THEN 2 WHEN 'mid' THEN 3
    WHEN 'small' THEN 4 WHEN 'tiny' THEN 5 END;

-- Q2: Does having no license correlate with risk?
SELECT
    CASE WHEN license = 'No license' THEN 'No license' ELSE 'Has license' END AS license_status,
    COUNT(*) AS total_repos,
    ROUND(100.0 * SUM(CASE WHEN risk_category = 'High' THEN 1 ELSE 0 END) / COUNT(*), 1) AS high_risk_pct,
    ROUND(AVG(stars), 0) AS avg_stars
FROM repos
GROUP BY license_status;

-- Q3: Does contributor count (bus factor) predict risk?
SELECT
    CASE
        WHEN distinct_recent_commit_authors <= 1 THEN '0-1'
        WHEN distinct_recent_commit_authors BETWEEN 2 AND 5 THEN '2-5'
        WHEN distinct_recent_commit_authors BETWEEN 6 AND 15 THEN '6-15'
        ELSE '16+'
    END AS contributor_bucket,
    COUNT(*) AS total_repos,
    ROUND(AVG(risk_score), 1) AS avg_risk_score,
    ROUND(100.0 * SUM(CASE WHEN risk_category = 'High' THEN 1 ELSE 0 END) / COUNT(*), 1) AS high_risk_pct
FROM repos
GROUP BY contributor_bucket
ORDER BY MIN(distinct_recent_commit_authors);

-- Q4: Risk by language (languages with 10+ repos)
SELECT
    language,
    COUNT(*) AS total_repos,
    ROUND(AVG(risk_score), 1) AS avg_risk_score,
    ROUND(100.0 * SUM(CASE WHEN risk_category = 'High' THEN 1 ELSE 0 END) / COUNT(*), 1) AS high_risk_pct
FROM repos
WHERE language != 'None/Not detected'
GROUP BY language
HAVING total_repos >= 10
ORDER BY avg_risk_score DESC;

-- Q5: Outlier check - open issues relative to recent commit activity
-- (Used to catch that AVG() was distorted by a few extreme repos;
--  the final version of this comparison uses medians, computed in pandas
--  because SQLite has no native MEDIAN.)
SELECT repo, risk_category, open_issues_total, commits_per_month_90d,
       ROUND(open_issues_total * 1.0 / (commits_per_month_90d + 0.1), 1) AS issues_per_activity
FROM repos
ORDER BY issues_per_activity DESC
LIMIT 10;
