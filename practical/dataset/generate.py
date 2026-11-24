#!/usr/bin/env python3
"""
Organizational Procedure Assistant — Training Dataset Generator

Generates JSONL training data for SFT fine-tuning of an LLM to classify
incoming organizational issues and respond with the company's exact procedures.

Usage:
    python generate.py

Outputs:
    train.jsonl      (~80% of examples)
    validation.jsonl (~20% of examples)
"""

import json
import os
import random

random.seed(42)

SYSTEM_PROMPT = (
    "You are an Organizational Procedure Assistant for a mid-sized software development company. "
    "When someone reports a situation, identify the issue type, state the severity, and provide "
    "the exact step-by-step procedure to follow — including escalation triggers and documentation requirements. "
    "Be specific, structured, and actionable."
)

# ---------------------------------------------------------------------------
# PROCEDURE RESPONSES — one canonical response per issue type
# These are the ground-truth outputs the model learns to produce.
# ---------------------------------------------------------------------------

PROCEDURES = {

"production_outage": {
"name": "Production Outage (P1)",
"response": """\
ISSUE TYPE: Production Outage (P1)
SEVERITY: Critical

IMMEDIATE (0–10 minutes):
1. Post in #incidents Slack channel immediately with timestamp and what is down
2. DevOps engineer claims ownership: "I have this"
3. AM sends holding message to client within 5 minutes:
   "We are aware of an issue affecting your system. Our team is actively investigating.
    We will update you in 30 minutes."
4. PM creates P1 ticket in Jira with exact timestamp

INVESTIGATION (10–40 minutes):
1. DevOps checks server health, uptime monitors, and error rate on dashboards
2. Tech Lead reviews last deployment — does timing match the incident start?
3. Check database connections, memory, and CPU on production servers
4. Review error tracking (Sentry/Datadog) for first occurrence timestamp
5. Isolate root cause: application code, infrastructure, or third-party dependency

ESCALATION TRIGGERS:
- 30 minutes unresolved → CTO joins the call immediately
- Suspected data loss or unauthorized access → switch to Security Incident procedure
- Multiple client projects affected → CEO notified

RESOLUTION:
1. Fix verified on staging environment before any production deployment
2. DevOps confirms service fully restored with health checks passing
3. AM sends resolution email: what happened, root cause, what was fixed, when
4. PM documents complete incident timeline in Jira ticket
5. Post-mortem scheduled within 48 hours of resolution
6. Post-mortem shared with client within 5 business days if requested

DOCUMENTATION REQUIRED:
- P1 incident ticket with full timeline and all decisions
- Root cause written in Confluence under the project space
- Monitoring alert added to prevent next occurrence if applicable\
"""},

"post_deploy_bug": {
"name": "Post-Deploy Bug (P2)",
"response": """\
ISSUE TYPE: Post-Deploy Bug (P2)
SEVERITY: High

IMMEDIATE (0–20 minutes):
1. Tech Lead confirms this is deployment-related by checking deploy time vs first report
2. Decision: rollback immediately or hotfix?
   - Rollback if it is clean and fast (no DB migrations involved)
   - Hotfix path if rollback carries risk
3. AM notifies client with scope of impact:
   "We have identified an issue with a recent deployment affecting [specific feature].
    Our team is working on a fix."

HOTFIX PATH:
1. Tech Lead creates hotfix branch from last stable tag
2. Developer applies minimal fix — no other changes allowed on this branch
3. QA Lead tests specifically for the reported issue plus regression on affected area
4. PM approves deploy after QA sign-off
5. DevOps deploys with Tech Lead present on call

ROLLBACK PATH:
1. DevOps rolls back to previous version immediately
2. PM confirms with client that service is restored
3. Tech Lead investigates root cause on rolled-back branch separately
4. Fix goes through full QA cycle before next deployment

ESCALATION TRIGGERS:
- Hotfix not ready within 2 hours → escalate to P1 procedure
- High-value client and significant impact → AM escalates to Account Director

RESOLUTION:
1. Fix confirmed working by QA on production
2. Client notified: what broke, what was fixed, when
3. Post-deploy test checklist reviewed: was this testable before deploy?
4. If yes → QA process gap logged and addressed in sprint retrospective

DOCUMENTATION REQUIRED:
- Jira ticket linked to the specific deployment
- Root cause and fix description in Confluence
- QA checklist updated if gap identified\
"""},

"security_incident": {
"name": "Security Incident / Data Breach",
"response": """\
ISSUE TYPE: Security Incident / Data Breach
SEVERITY: Critical

IMMEDIATE (0–30 minutes) — CONTAIN FIRST, DO NOT DELETE ANYTHING:
1. CTO notified immediately — takes command of the response
2. Affected systems ISOLATED from the network — preserve all logs
3. No external communication yet — CEO must approve all messaging
4. Legal/Compliance officer engaged immediately

ASSESSMENT (30 minutes–2 hours):
1. CTO and DevOps determine:
   - What data was accessed and how much?
   - Is the breach ongoing or contained?
   - Which systems and credentials are compromised?
2. CEO briefed with confirmed facts only — no speculation

CLIENT NOTIFICATION:
- Under GDPR: notify within 72 hours of confirmed breach
- AM drafts notification — Legal and CEO must approve before it is sent
- Notification includes: what happened, what data, what we are doing
- Never speculate in writing — only confirmed facts

REGULATORY:
- EU client data: GDPR notification to supervisory authority within 72 hours
- HIPAA-covered data: notify within 60 days
- Legal officer manages all regulatory communication

REMEDIATION:
1. Patch confirmed vulnerability before bringing systems back online
2. Force password reset for all affected accounts
3. Audit all access logs for 30 days prior to incident
4. External security review if breach was sophisticated
5. Penetration test before returning system to full operation

ESCALATION TRIGGERS:
- Any confirmed unauthorized data access → CEO and Legal immediately
- Media or regulatory inquiry received → Legal manages all responses

DOCUMENTATION REQUIRED:
- Incident logged in ISO 27001 security incident register
- Root cause documented within 72 hours
- Corrective action plan submitted within 7 days
- All communication records preserved for potential legal use\
"""},

"client_escalation": {
"name": "Client Escalation / Relationship at Risk",
"response": """\
ISSUE TYPE: Client Escalation / Relationship at Risk
SEVERITY: High

IMMEDIATE (within 2 hours of complaint):
1. AM acknowledges to client — do not be defensive:
   "Thank you for raising this. I take this seriously and will come back
    to you by [specific time] today."
2. AM briefs Account Director and PM immediately
3. Single point of contact rule: all client communication goes through AM only

INTERNAL ASSESSMENT:
1. PM pulls all relevant communication history and deliverables
2. PM and Tech Lead review what was promised vs what was delivered
3. Honest internal answer: is the client right?
   - If yes → own it, fix it, offer specific remedy
   - If no → prepare clear factual response with evidence from SOW and history
   - If partial → acknowledge what is valid, address what is not

RESPONSE CALL:
1. Account Director (not AM alone) leads the client call
2. CTO joins if the complaint is technical
3. Listen fully before responding — do not interrupt or be defensive
4. Agree on specific actions with specific deadlines before the call ends
5. Send written summary of agreed actions within 24 hours

ESCALATION TRIGGERS:
- Client threatens to cancel contract → CEO joins next call immediately
- Written legal threat received → Legal Officer engaged, no further direct response
- Do not make financial commitments without CEO approval

RESOLUTION:
1. Agreed actions tracked in Jira with daily progress updates to AM
2. AM sends progress update to client every 2 days until fully resolved
3. PM documents full incident in project history in Confluence
4. 2-week follow-up call after resolution to confirm satisfaction

DOCUMENTATION REQUIRED:
- All communication logged with timestamps
- Agreed actions and resolution documented in project history
- If financial remedy offered: Finance and Legal must review before confirming\
"""},

"scope_change": {
"name": "Scope Change / Out-of-Scope Request",
"response": """\
ISSUE TYPE: Scope Change / Out-of-Scope Request
SEVERITY: Medium

IMMEDIATE (same business day):
1. PM acknowledges to client — positive tone, zero commitment:
   "Great idea — let us assess this properly and come back to you with an estimate."
2. NEVER say yes to scope without internal assessment and a signed CR
3. NEVER say no without offering the Change Request path

INTERNAL ASSESSMENT:
1. PM confirms the requested feature is NOT in the signed SOW
2. Tech Lead estimates effort in hours
3. PM calculates cost at the client's contracted rate
4. PM assesses impact on current timeline and next sprint

CHANGE REQUEST DOCUMENT — PM prepares formal CR containing:
- Description of the requested change
- Hours estimated
- Cost in USD
- Timeline impact: additional days OR fits within next sprint
- Dependencies or risks
- Client signature required before any work begins

CLIENT PRESENTATION:
1. AM presents CR to client — frame as adding value, not charging extra
2. If client disputes the item is out of scope → PM provides exact SOW clause reference
3. If genuinely ambiguous → CTO and PM decide internally, then client discussion

ESCALATION TRIGGERS:
- Client refuses to pay but demands the work → escalate to Account Director
- Dispute exceeds $5,000 USD → Legal reviews the SOW before any commitment
- Do NOT start work until CR is signed — no exceptions

DOCUMENTATION REQUIRED:
- Signed CR filed in project documentation folder
- Jira ticket created linked to the CR reference number
- Sprint plan updated to reflect added scope\
"""},

"deadline_at_risk": {
"name": "Project Deadline at Risk",
"response": """\
ISSUE TYPE: Project Deadline at Risk
SEVERITY: High

IDENTIFICATION — PM must flag as soon as the risk is known, not at the deadline:
1. PM updates Jira with a risk flag immediately
2. Rule: if there is more than 40% chance of missing the deadline, flag it now
3. Waiting until the deadline has passed is a process failure

INTERNAL MEETING (within 24 hours of identifying risk):
1. PM, Tech Lead, and relevant developers assess:
   - What is not done and why?
   - Was this underestimated, out-of-scope growth, blocker, or absence?
   - What can be cut to still deliver core features on time?
   - Can resource be added?
2. Options ranked: reduce scope → extend timeline → add resource
3. CTO makes final call if engineers disagree

CLIENT COMMUNICATION — tell the client BEFORE the deadline, never after:
1. AM informs client proactively:
   "We want to give you advance notice that [feature X] is running slightly
    behind our original estimate due to [reason]."
2. Present one clear recommendation — not three options for the client to solve
3. Provide a specific new date with a confidence level
4. Never blame a team member or vendor in client communication

ESCALATION TRIGGERS:
- Deadline is contractual and will be missed → Legal reviews liability immediately
- Client refuses any extension → Account Director escalates to CEO

RESOLUTION:
1. PM logs root cause in project retrospective
2. Lessons learned added to PM process handbook
3. All future estimates for this client type revised if systemic

DOCUMENTATION REQUIRED:
- Risk flag in Jira with date first identified
- Internal assessment notes
- Client communication record with timestamps\
"""},

"key_resource_leaving": {
"name": "Key Developer Leaving Mid-Project",
"response": """\
ISSUE TYPE: Key Developer Leaving Mid-Project
SEVERITY: High

DAY 1 — CONTAINMENT:
1. HR conducts immediate conversation: confirm last working date
2. PM informed same day — assess project impact immediately
3. Access NOT revoked yet — transition must come first
4. CTO identifies replacement resource within 48 hours

WEEK 1 — KNOWLEDGE TRANSFER:
1. Developer documents everything they own:
   - Architecture decisions and the reasoning behind them
   - Any credentials or config they hold personally
   - Open tickets they understand best
   - Known issues not yet in Jira
2. All credentials transferred to company password manager immediately
3. GitHub, AWS, and database access moved to shared team accounts
4. Replacement developer assigned and introduced to the project

WEEK 2 — SHADOWING:
1. Replacement joins all client calls as observer
2. Replacement reviews all major PRs from the last 3 months
3. Pair programming sessions — departing developer and replacement together
4. Tech Lead reviews handover document completeness before last day

LAST DAY:
1. HR exit interview
2. All access revoked within 2 hours of end of working day:
   GitHub, Jira, Slack, AWS, production databases, email, VPN
3. Equipment returned and wiped by IT

CLIENT:
- If developer was visible to the client → AM informs them proactively
- Introduce replacement before the developer leaves, not after
- Frame positively: continuity assured, no disruption expected

SUDDEN DEPARTURE (no notice):
1. CTO locks all access within 1 hour
2. Emergency audit: what systems did this person have access to?
3. Rotate all shared secrets and passwords immediately
4. Treat as potential security incident until confirmed otherwise

DOCUMENTATION REQUIRED:
- Handover document signed off by Tech Lead
- Access revocation checklist completed and filed by IT\
"""},

"invoice_overdue": {
"name": "Invoice Overdue / Client Not Paying",
"response": """\
ISSUE TYPE: Invoice Overdue / Client Not Paying
SEVERITY: Medium → High (escalates with time)

DAY 15 (first overdue):
- Finance sends automated payment reminder — polite and factual
- No escalation yet

DAY 30:
1. Finance notifies AM of overdue invoice with full invoice details
2. AM contacts client informally:
   "Just checking in on invoice #X — do you need anything from us to process it?"
3. All communication documented with timestamps

DAY 45:
1. Account Director sends formal written notice
2. Work on new features paused — maintenance and critical support continues
3. Client informed of this pause in writing
4. PM notified: do not start any new sprint until payment is received

DAY 60:
1. Legal/Compliance reviews contract payment terms
2. Formal notice of default sent via certified email
3. All new work stopped
4. CTO and CEO briefed

DAY 75+:
1. CEO makes final decision: payment plan, collection agency, or legal action
2. If legal action → Legal Officer engages solicitor
3. Client relationship put on hold pending resolution

ESCALATION TRIGGERS:
- Client becomes unresponsive at any stage → escalate to next level immediately
- Client disputes the invoice → Legal reviews contract before any response

PREVENTION (applied to all new clients):
- 50% upfront deposit before work starts
- Milestone billing: invoice tied to deliverable, not calendar date
- Late payment clause in all contracts: interest accrues per month after due date

DOCUMENTATION REQUIRED:
- All payment communication logged with dates and amounts
- Formal notices filed in client contract folder\
"""},

"client_onboarding": {
"name": "New Client Onboarding / Project Kickoff",
"response": """\
ISSUE TYPE: New Client Onboarding / Project Kickoff
SEVERITY: N/A — Standard Process

PRE-KICKOFF (3–5 business days before kickoff call):
1. Finance confirms deposit received — nothing starts before this
2. AM introduces PM to client via email
3. PM reads signed SOW completely — flags any ambiguities to AM
4. Tech Lead reviews technical requirements and identifies risks
5. PM sets up:
   - Jira project with initial epics from SOW
   - Confluence space with project documentation
   - Slack channel (#client-name-project)
   - GitHub repository with branch protection
   - Shared folder for client assets and design files

KICKOFF CALL AGENDA:
1. Introductions — who does what on both sides
2. Project scope walkthrough — confirm shared understanding
3. Communication norms: how, when, and with whom
4. Tools and access: what the client will have visibility into
5. First milestone and delivery date confirmed and agreed
6. Questions from both sides addressed

POST-KICKOFF (within 48 hours):
1. PM sends meeting notes and agreed decisions to client in writing
2. Dev environment provisioned by DevOps
3. Client granted Jira access (view only by default)
4. Tech Lead creates initial technical architecture document
5. QA Lead sets up test environment
6. Sprint 0 planned: environment setup, CI/CD pipeline, boilerplate

FIRST SPRINT REVIEW (end of sprint 1):
- Client sees working software — even if basic — from day one
- Feedback loop established immediately
- Any scope concerns surfaced and logged as early as possible

DOCUMENTATION REQUIRED:
- Kickoff notes with agreed decisions sent to client
- Project setup checklist completed and filed
- Technical architecture document draft in Confluence\
"""},

"performance_issue": {
"name": "Performance Issue / Application Running Slow",
"response": """\
ISSUE TYPE: Performance Issue / Application Running Slow
SEVERITY: Medium (High if SLA is breached)

IDENTIFICATION:
1. Source: client complaint, monitoring alert, or internal QA finding
2. PM logs performance ticket in Jira with specific metrics if available
3. Severity assessed: affects all users? Core features? SLA breach?

INVESTIGATION:
1. DevOps reviews server metrics: CPU, memory, network I/O over last 24 hours
2. Tech Lead reviews: slow query logs, N+1 query patterns, missing database indexes
3. Check: did degradation start after a specific deployment or data growth milestone?
4. Profiling run on the slowest endpoints
5. Check CDN, caching layers, and external API response times

ROOT CAUSES — investigate in this order:
1. Database: missing index, slow query, table lock, connection pool exhaustion
2. Application: N+1 query pattern, memory leak, unoptimised loop or algorithm
3. Infrastructure: undersized server, no caching layer, missing CDN
4. External: third-party API slow (Stripe, Twilio, SendGrid, auth provider)

FIX PRIORITY:
1. Quick wins first: add index, enable caching, optimize obvious queries
2. Application changes: query optimization, pagination, lazy loading
3. Infrastructure scaling: last resort — treats symptom, not root cause

CLIENT COMMUNICATION:
- SLA breached → AM notifies client same day with ETA
- Not SLA breach → fix quietly, mention improvement in next sprint update
- Never promise a specific performance number without testing it first

ESCALATION TRIGGERS:
- SLA breach and fix not ready within agreed window → escalate to CTO
- Multiple users unable to use the product → escalate to P2 procedure

DOCUMENTATION REQUIRED:
- Performance ticket with before/after metrics
- Root cause and fix documented in Confluence
- Monitoring threshold updated to catch this earlier next time\
"""},

"qa_failure": {
"name": "QA Failure / Client Rejects Deliverable",
"response": """\
ISSUE TYPE: QA Failure / Client Rejects Deliverable
SEVERITY: Medium → High

INTERNAL QA FAILURE (caught before delivery):
1. QA Lead logs all defects in Jira with severity classification (P1/P2/P3)
2. P1 and P2 defects → delivery delayed until resolved — no exceptions
3. P3 defects → PM decides: fix now or deliver with known issues disclosed?
4. Tech Lead assigns defects and sets fix timeline
5. QA re-tests after every fix before any delivery attempt

CLIENT REJECTION (after delivery):
1. PM reviews each rejection item within the same business day:
   - Genuine bug? → fix immediately, no debate
   - Scope interpretation difference? → PM reviews SOW clause by clause
   - New requirement disguised as a bug? → Change Request path
2. AM responds to client within 4 hours:
   "Thank you for the detailed feedback. We are reviewing each point
    and will respond by [specific time]."
3. PM provides written response on each item — bug acknowledgment or scope discussion

REPEAT REJECTION (same feature rejected twice):
1. Tech Lead and QA Lead review the full process:
   - Were requirements documented before development started?
   - Was QA involved in requirement review?
   - Was a prototype or wireframe shown before full build?
2. Process gap documented and corrected before next sprint

ESCALATION TRIGGERS:
- Client refuses all deliverables despite meeting spec → Account Director involved
- Written contract dispute → Legal review immediately

DOCUMENTATION REQUIRED:
- Defect log with resolution timestamps in Jira
- Written response to client rejection items on file
- Process improvement notes in QA handbook if repeat rejection occurred\
"""},

"hr_issue": {
"name": "HR Issue / Team Conflict",
"response": """\
ISSUE TYPE: HR Issue / Team Conflict
SEVERITY: Medium → High

REPORT RECEIVED:
1. HR Manager acknowledges to reporter within 24 hours
2. Reporter's identity kept confidential — no exceptions under any pressure
3. HR Manager does NOT share details with anyone not directly involved
4. Informal resolution attempted first if the situation is appropriate for it

INVESTIGATION:
1. HR Manager speaks separately with all parties involved
2. Line manager NOT included in HR conversation if they are part of the dispute
3. Facts documented in writing in HR file — not in Jira or Confluence
4. HR assesses: informal resolution, formal process, or immediate action

CONFLICT BETWEEN PEERS — informal path:
1. HR facilitates structured mediation session with both parties
2. Agreed working norms documented and signed by both
3. Follow-up check-in with both parties in 2 weeks
4. If mediation fails → formal process begins

FORMAL COMPLAINT (harassment, discrimination, serious misconduct):
1. HR Manager briefs CEO immediately — same day
2. External HR legal advisor engaged before any formal response
3. Parties separated where operationally possible pending investigation
4. Formal investigation process with documented evidence chain
5. Outcome communicated to both parties in writing

LEADERSHIP INVOLVED:
- CTO or senior manager subject of complaint → CEO leads the investigation
- CEO involved → Board notified

ESCALATION TRIGGERS:
- Any allegation of harassment or discrimination → formal process immediately
- Safety concern raised → immediate action, same day

DOCUMENTATION REQUIRED:
- All HR conversations documented by HR Manager
- Outcomes stored in HR system only — never in Confluence or Jira
- ISO 27001: HR records treated as strictly confidential personal data\
"""},

}

# ---------------------------------------------------------------------------
# INPUT MESSAGES — varied phrasings for each issue type
# Each list covers: direct, vague, urgent, technical, frustrated, polite
# ---------------------------------------------------------------------------

MESSAGES = {

"production_outage": [
    # Direct
    "Production is completely down. All users getting errors. Started about 20 minutes ago.",
    "The platform is unavailable. Clients can't login. We need someone on this now.",
    "Full outage on the main application. No one can access anything.",
    "Server is down. All endpoints returning errors. This just started.",
    "Complete service outage — 502 errors on every request.",
    # Vague
    "Something's wrong with the app. Users are saying they can't get in?",
    "Hey, the site seems to be down? Getting reports from multiple clients.",
    "Not sure what's happening but the application isn't loading for anyone.",
    "We're getting complaints. I think the server might be having issues.",
    "The system might be down? People are saying they can't access it.",
    # Urgent / stressed
    "URGENT — production is completely down. The client is calling every 5 minutes.",
    "CRITICAL — everything is offline! This is a major client, need help immediately!",
    "Emergency! The whole platform went down 30 minutes ago and I haven't been able to reach anyone!",
    "This is a P1. The entire production environment is unavailable right now.",
    "Site is down and the client CEO just emailed me personally. We need this fixed NOW.",
    # Technical
    "ECS service health checks all failing. Load balancer returning 502. Started at 14:30.",
    "Database connection pool is exhausted. All requests timing out at the application layer.",
    "Memory usage hit 100% on prod. Node process crashed and won't restart cleanly.",
    "Nginx returning 502 Bad Gateway. Upstream gunicorn workers not responding.",
    "Redis cluster went down and the application doesn't have fallback. Everything is broken.",
    # Frustrated / emotional
    "I cannot believe this is happening again. The site has been down for an hour.",
    "This is unacceptable. Production has been down since this morning and no one told me.",
    "Why is the site down AGAIN? Clients are losing their minds.",
],

"post_deploy_bug": [
    # Direct
    "We deployed 2 hours ago and now the checkout flow is broken. Users can't complete orders.",
    "The deployment from this afternoon broke the user authentication. People are getting logged out randomly.",
    "Post-deploy issue — the reporting module stopped working after today's release.",
    "Bug introduced in today's deployment. Payment processing is failing intermittently.",
    "Since the last release, the file upload feature is completely broken.",
    # Vague
    "Something broke after the deploy today. Users are complaining but I'm not sure exactly what.",
    "The deployment seems to have caused some issues. Getting reports of things not working.",
    "I think the release broke something. Can someone check?",
    "After today's update, some features aren't working properly it seems.",
    "The app is behaving strangely since we deployed this morning.",
    # Urgent
    "The deploy just went out 10 minutes ago and already 20 users have reported the login is broken. URGENT.",
    "Critical bug from today's deployment — the billing system is broken and clients can't pay.",
    "Emergency rollback needed. The deployment we just did has broken the core feature.",
    # Technical
    "The deploy introduced a null pointer in the order service. Orders created after 15:00 are failing.",
    "Migration ran but it dropped a column that the new code still references. Getting SQL errors.",
    "The build included a package upgrade that changed the JWT signature algorithm. All tokens invalid.",
    "New code is trying to call a deprecated API endpoint. Getting 410 errors on all profile updates.",
    # Frustrated
    "This is the third time a deploy has broken something this month. We need to fix our process.",
    "Who approved this deployment? The login is completely broken and no one tested it.",
],

"security_incident": [
    # Direct
    "We may have had a data breach. Unauthorized access detected on our production database.",
    "Security alert: suspicious login activity on our admin accounts from unknown IPs.",
    "Client reported they received a phishing email that appears to use our system data.",
    "Our monitoring flagged unauthorized API access. Someone is querying our database externally.",
    "We found evidence that someone accessed our codebase without authorization.",
    # Vague
    "Something weird is happening with our servers. Unusual traffic patterns in the logs.",
    "A client just told me someone called them pretending to be from our company. Could be related to a breach.",
    "I noticed some accounts that shouldn't exist in the admin panel. Might be a security issue?",
    "Getting alerts I don't understand — might be someone trying to get into our systems.",
    "One of our developers reported their credentials might have been stolen.",
    # Urgent / serious
    "BREACH — unauthorized user has been accessing client data. We need to act immediately.",
    "Critical security incident. We have confirmed unauthorized access to production systems.",
    "Data leak suspected. Client data may have been exposed. Need immediate response.",
    # Technical
    "Unusual query patterns in DB logs: 50,000 SELECT queries from an IP we don't recognize starting at 02:00.",
    "SSH brute force on our bastion host. Multiple failed attempts then a successful login from an unknown IP.",
    "S3 bucket was misconfigured as public. It contained client contract PDFs. Bucket is now private.",
    "API key was committed to a public GitHub repo by mistake. It was live for 3 hours before we caught it.",
],

"client_escalation": [
    # Direct
    "Client is very unhappy and threatening to cancel the contract. They want to speak to someone senior.",
    "Received a formal complaint from a client about delays and communication quality.",
    "Client relationship is at risk. They've told us they're evaluating other vendors.",
    "Major client sent a formal letter of complaint about our delivery quality.",
    "Client escalated to our CEO directly. They are very unhappy with the project progress.",
    # Vague
    "Client seems upset but I'm not sure exactly what the issue is. The last call didn't go well.",
    "The client has gone cold — not responding to emails. Something is wrong.",
    "One of our accounts might be in trouble. The client was quite hostile on the last call.",
    "Getting a bad feeling about the client situation. They were very critical in today's meeting.",
    "Client asked to pause the project. I think they might be unhappy but won't say directly.",
    # Urgent
    "CLIENT IS ESCALATING TO THEIR BOARD. We need a response today.",
    "Client just sent a termination notice. We have 24 hours to respond.",
    "Emergency — major client is about to cancel. I need senior leadership on a call today.",
    # Polite but serious
    "One of our most valuable clients has raised concerns about the project. I'd like to escalate internally.",
    "I need to flag a client relationship issue before it gets worse. They've raised concerns twice now.",
    # Frustrated
    "I've been trying to handle this client situation alone for weeks and it's getting worse. I need help.",
    "The client is right to be unhappy. We've let them down and I need support to fix this.",
],

"scope_change": [
    # Direct
    "Client is asking for a new feature that is definitely not in the original SOW.",
    "Client wants to add a full reporting module. That's out of scope but they don't seem to know that.",
    "New request from client for mobile app integration — the contract is for web only.",
    "Client sent a list of 10 new features they want added before launch. None of them were agreed.",
    # Vague
    "Client mentioned wanting some extra things done. Not sure if that's in scope or not.",
    "The client keeps asking for 'small changes' that are starting to add up.",
    "Client called about some additional work they need. I need to check if it's covered.",
    "We're getting feature requests that I don't think were in the original plan.",
    # Polite / tricky
    "Client is asking us to 'quickly add' a feature that would take at least 2 weeks to build properly.",
    "Client mentioned they assumed X was included. It definitely wasn't in the SOW but they seem genuine.",
    "The client is asking for something new but phrasing it as if it's part of what we agreed.",
    # Technical
    "Client wants us to add real-time WebSocket functionality. The spec was for REST API only.",
    "Client is asking for a machine learning recommendation engine. The project was basic CRUD.",
    "Client wants multi-language support added. Wasn't in requirements, would require full refactor.",
    # Difficult
    "Client is demanding extra work for free, saying it was 'implied' in the original agreement.",
    "Client is refusing to pay for a Change Request, saying we should have known they needed this.",
],

"deadline_at_risk": [
    # Direct
    "We're not going to hit the milestone deadline next Friday. Flagging this now.",
    "Sprint 4 is running 1 week behind. The delivery date is at risk.",
    "The team is behind on the current phase. We need to discuss options before the deadline.",
    "Development is taking longer than estimated. We won't meet the agreed date.",
    # Vague
    "Things aren't going as planned. I think we might have a timing issue coming up.",
    "The team seems stretched. I'm worried about the upcoming deadline.",
    "Not sure we're going to make it. Progress has been slower than expected.",
    "Deadline might be in trouble. The estimate wasn't quite right.",
    # Urgent
    "The delivery is in 5 days and we're at least 2 weeks of work away. We need a plan now.",
    "Critical timeline issue — the contractual milestone is next week and we're way behind.",
    # With reason
    "One of our senior developers was sick for 2 weeks and we're now behind on the backend work.",
    "The API we depend on changed their interface. We've lost 1.5 weeks reworking the integration.",
    "Requirements changed mid-sprint and now the original estimate is wrong.",
    "Third-party vendor delayed their delivery and we can't proceed without it.",
    # Reflective
    "I think we underestimated this sprint. We need to have an honest conversation about the timeline.",
    "The original estimate was optimistic. We need to reset client expectations now.",
],

"key_resource_leaving": [
    # Direct
    "Our lead developer just handed in his resignation. He's leaving in 2 weeks.",
    "One of our senior engineers has resigned. She was the only person who understood the payment system.",
    "Key developer is leaving next month. This project has no one else who knows the architecture.",
    "Tech Lead just gave notice. Mid-project. This is a major risk.",
    # Sudden
    "Developer left without notice. We found out today. We have no access to his credentials.",
    "Our developer was let go today and we have an active client project. What do we do?",
    "Engineer resigned effective immediately. She had access to production and all the client accounts.",
    # Vague
    "We might be losing a developer soon. Not confirmed yet but I need to plan for it.",
    "One of the team is thinking about leaving. We need a contingency plan for the project.",
    # With context
    "Our only mobile developer resigned. The client's app is halfway through development.",
    "The developer who built the entire backend just quit. No documentation exists.",
    "Team member leaving next week. They're the only one who knows how the legacy integration works.",
    # Worried
    "We're in the middle of a critical delivery and just lost our best engineer. How do we handle this?",
    "This resignation couldn't have come at a worse time. We're 3 weeks from delivery.",
],

"invoice_overdue": [
    # Direct
    "Invoice #1047 is 45 days overdue. Client is not responding to Finance.",
    "The client hasn't paid the milestone invoice from last month. We need to escalate.",
    "Payment is 60 days late. The client keeps saying 'it's being processed'.",
    # Vague
    "Client hasn't paid yet. It's been a while. Not sure what's happening.",
    "Finance is chasing an overdue invoice but I'm not sure when to escalate.",
    "Payment seems delayed. Should I be worried?",
    # Firm
    "Invoice is now 75 days overdue. We've sent 4 reminders. No payment received.",
    "Client is 90 days past due. We need to decide whether to stop work.",
    "Client disputes the invoice but won't explain why. This has been going on for 2 months.",
    # Tricky
    "Client says they're having cash flow issues and asked for a payment plan. What should we do?",
    "Client wants to continue the project but still hasn't paid the previous invoice.",
    "New contact at the client says they never approved the invoice and won't pay it.",
    # Context
    "This is a small client, the invoice is only $3,000, but it's been 60 days with no response.",
    "Long-term client, never had payment issues before, now 45 days overdue. Something seems off.",
],

"client_onboarding": [
    # Direct
    "New client contract just signed. We need to kick off the project. What's the process?",
    "Contract is signed and deposit is received. How do we officially start onboarding?",
    "New project starting next week. What do I need to set up before the kickoff call?",
    # Practical
    "Need to set up Jira, Confluence, Slack, and GitHub for a new client. Any checklist I should follow?",
    "First meeting with new client is scheduled for Thursday. What needs to happen before that?",
    "Sales just closed a new client. I'm the PM assigned. What are the first steps?",
    # Questions
    "When should we send the client their project credentials? Before or after kickoff?",
    "Should the client have access to Jira from day one or only after we set up the structure?",
    "New client asked for a meeting tomorrow to 'get started'. Contract was only signed today.",
    # Context
    "This is a returning client starting a new project. Same contacts, different scope. Normal process?",
    "Client is based in the UK. Any different process or considerations for the kickoff?",
    "This is our biggest client to date. Anything extra we should do for the onboarding?",
],

"performance_issue": [
    # Direct
    "Client is reporting the application is running very slowly. Page load times are 10+ seconds.",
    "API response times have degraded significantly. Everything is taking 3-5x longer than usual.",
    "Users are complaining about the dashboard loading slowly. It's affecting their daily work.",
    # Vague
    "The app feels slow. Not sure if it's our side or the client's internet.",
    "Getting complaints about performance. Unclear where the bottleneck is.",
    "Something's making the app sluggish. Response times seem off.",
    # Technical
    "Database query time went from 50ms to 2000ms after the data grew past 1 million rows. Missing index?",
    "The reports page is timing out for large date ranges. Query is scanning the full table.",
    "Memory usage climbing over 24 hours and the app slows down. Probably a memory leak.",
    "Third-party payment API is slow (avg 8 seconds) and it's blocking our checkout flow.",
    # SLA at risk
    "We're breaching the SLA response time. 99th percentile is 8 seconds, our agreement says under 3.",
    "Client's SLA monitoring caught us. Average response time is 4.5 seconds, limit is 2 seconds.",
    # Worried
    "Performance has been slowly degrading for weeks. I think we need to address this properly.",
    "Client mentioned performance 3 times in the last 2 meetings. We need to take this seriously.",
],

"qa_failure": [
    # Direct
    "Client rejected the sprint delivery. They've listed 12 issues with the new features.",
    "QA caught critical bugs before delivery. We can't ship this week as planned.",
    "Client sent back the deliverable with significant feedback. Several features don't work as expected.",
    # Vague
    "Client wasn't happy with the demo. Not sure if it's bugs or expectations.",
    "QA is finding issues. Not sure if we can deliver on time.",
    "Client feedback was negative. They seem unhappy with the quality.",
    # Specific
    "The client rejected the invoicing module. The PDF export format doesn't match their requirement at all.",
    "QA found a data loss bug in production — if you delete a record it also deletes related records.",
    "Client says the mobile layout is broken on iOS. We only tested Android.",
    # Process issue
    "This is the second sprint in a row the client has rejected. Something systemic is wrong.",
    "Client keeps finding bugs we should have caught internally. Our QA process is failing.",
    "We delivered something that didn't match the wireframes. How did this get through QA?",
    # Frustrated
    "Client sent a detailed bug report with screenshots. I'm embarrassed by some of these issues.",
    "The client's own QA found more bugs in 2 hours than our team found in a week of testing.",
],

"hr_issue": [
    # Direct
    "Two team members are in conflict and it's affecting the whole team's work.",
    "An employee has filed a formal complaint against their line manager.",
    "We have a conduct issue on the team. Need HR involved immediately.",
    # Vague
    "There's tension on the team. Something happened but I don't know the full story.",
    "One of the developers seems really unhappy. I'm worried about the team dynamics.",
    "Something is off with the team. Morale is very low and I think there's an underlying issue.",
    # Serious
    "An employee has reported harassment by a colleague. This needs to be handled formally.",
    "Formal complaint received regarding discrimination. This is serious and needs immediate attention.",
    "Team member raised a serious allegation. I need to know the right process.",
    # Manager issue
    "Several team members have come to me separately with concerns about the same team lead.",
    "An employee says their manager is bullying them. What's the process?",
    # Practical
    "Two developers refuse to work together after a disagreement. It's disrupting the project.",
    "There was a heated argument in the office today. Both parties are upset. What do we do?",
    "Team member sent an aggressive email to a colleague. HR needs to know about this.",
],

}

# ---------------------------------------------------------------------------
# EDGE CASES — ambiguous messages that span two issue types
# The model must decide the primary issue and handle both aspects.
# ---------------------------------------------------------------------------

EDGE_CASES = [
    {
        "input": "Client is furious because the site went down during their product launch. They're threatening to cancel.",
        "output": """\
ISSUE TYPE: Production Outage (P1) [with concurrent Client Escalation]
SEVERITY: Critical

Handle in this order: resolve the outage first, then address the relationship.

IMMEDIATE — OUTAGE (0–10 minutes):
1. DevOps claims ownership and begins investigation immediately
2. AM sends holding message to client acknowledging the timing issue
3. CTO notified immediately given the client escalation risk

PARALLEL — CLIENT ESCALATION:
1. Account Director notified of escalation risk alongside the outage
2. No defensive response to client yet — resolve the technical issue first
3. CEO briefed given cancellation threat

POST-OUTAGE — RELATIONSHIP RECOVERY:
1. Account Director leads a recovery call once service is restored
2. Acknowledge the impact of the timing specifically — do not minimise it
3. Agree on a remedy or goodwill gesture with CEO approval before offering
4. Post-mortem shared with client proactively

DOCUMENTATION REQUIRED:
- P1 incident ticket with timeline
- Client escalation log with all communication
- Agreed remediation actions in writing\
"""
    },
    {
        "input": "The developer who built our deployment system just resigned and now production is down.",
        "output": """\
ISSUE TYPE: Production Outage (P1) [complicated by Key Resource Departure]
SEVERITY: Critical

Handle in this order: restore production first, then manage the resource situation.

IMMEDIATE — OUTAGE (0–10 minutes):
1. All available engineers mobilized — DevOps and Tech Lead immediately
2. If the departed developer's knowledge is needed, attempt to contact them
   as a professional courtesy — do not assume they will help
3. AM holds the client with standard P1 messaging
4. CTO joins immediately — this is high risk

IMMEDIATE — ACCESS SECURITY:
1. Lock the departed developer's access to ALL systems within 1 hour
2. Audit what they had access to — this is now a security hygiene requirement
3. Rotate any credentials they may have held

DOCUMENTATION — POST-RESOLUTION:
1. Document everything the departed developer knew that was not written down
2. Single points of knowledge are a critical process risk — log it
3. CTO to review bus factor across all projects

DOCUMENTATION REQUIRED:
- P1 incident record
- Access revocation checklist completed
- Knowledge gap assessment for affected systems\
"""
    },
    {
        "input": "Client is asking for new features and also says the current app is too slow. Where do we start?",
        "output": """\
ISSUE TYPE: Performance Issue + Scope Change Request (dual issue)
SEVERITY: Medium

Handle the performance issue first — it is a delivery obligation. The new features are a Change Request.

IMMEDIATE — PERFORMANCE:
1. Tech Lead and DevOps assess the performance complaint with specific metrics
2. PM logs a performance ticket in Jira
3. If SLA is breached → notify client of investigation and ETA immediately
4. If not SLA breach → fix in current sprint before accepting new scope

SCOPE CHANGE — NEW FEATURES:
1. PM acknowledges feature request with no commitment:
   "Let us resolve the current performance issue first, then we will assess
    the new features and come back to you with an estimate."
2. Do NOT start work on new features until performance is addressed and a CR is signed
3. Formal Change Request document prepared once performance is stable

DOCUMENTATION REQUIRED:
- Performance ticket with metrics
- Formal CR for new features once agreed\
"""
    },
    {
        "input": "Team member raised concerns about their manager and also said they're thinking of leaving.",
        "output": """\
ISSUE TYPE: HR Issue / Team Conflict [with Key Resource Retention risk]
SEVERITY: High

Handle the HR concern formally first. Retention is secondary and must not compromise the HR process.

IMMEDIATE — HR:
1. HR Manager acknowledges to the employee within 24 hours
2. HR conversation is confidential — manager is not informed yet
3. HR assesses: informal resolution or formal complaint?
4. All conversation documented in HR file only

PARALLEL — RETENTION RISK:
1. HR Manager briefs CEO on retention risk in general terms without compromising confidentiality
2. CTO notified of potential resource risk for project planning only
3. No promises made to the employee that bypass the HR process

POST-HR RESOLUTION:
1. If HR issue is resolved → retention conversation handled separately by appropriate manager (not the subject of the complaint)
2. CTO assesses project risk and identifies contingency resourcing
3. If employee decides to leave → standard departure procedure follows

DOCUMENTATION REQUIRED:
- HR file (confidential)
- CTO notified of resource risk (no HR details shared)\
"""
    },
    {
        "input": "Invoice is 60 days overdue and the client is also requesting a large scope change.",
        "output": """\
ISSUE TYPE: Invoice Overdue [blocks Scope Change discussion]
SEVERITY: High

Do not discuss new scope until payment is resolved. These are handled sequentially, not in parallel.

IMMEDIATE — PAYMENT:
1. Finance and Account Director: formal follow-up on the overdue invoice
2. All new feature discussion paused explicitly:
   "We would love to discuss the new scope with you. To move forward effectively,
    we need to resolve the outstanding invoice first."
3. No new work starts under any circumstances until payment is received

SCOPE CHANGE — ON HOLD:
1. PM logs the scope request but does NOT assess or estimate it yet
2. Scope discussion resumes only after payment is received in full
3. If client pays → standard Change Request process begins immediately

ESCALATION:
- Client refuses to pay but demands new work → Account Director and Legal involved
- Relationship at risk → CEO briefed

DOCUMENTATION REQUIRED:
- Invoice communication log
- Written communication to client that new scope is on hold pending payment\
"""
    },
    {
        "input": "Security alert fired but it might just be a false positive. Should we still escalate?",
        "output": """\
ISSUE TYPE: Security Incident / Data Breach (treat as real until proven otherwise)
SEVERITY: Critical — assumption of real incident until disproven

IMMEDIATE — ALWAYS ESCALATE A SECURITY ALERT:
1. CTO notified immediately — do not resolve this alone
2. Isolate potentially affected systems while investigating — do not wait for confirmation
3. Preserve all logs before any investigation actions
4. CEO briefed as a precaution

INVESTIGATION:
1. CTO and DevOps determine: what triggered the alert, when, and from where?
2. Check: is there any evidence of actual access or data movement?
3. Review access logs for the full 24-hour period around the alert
4. Only CTO can declare a confirmed false positive — not anyone else

IF CONFIRMED FALSE POSITIVE:
1. Document exactly what triggered the alert and why
2. Adjust monitoring threshold if the alert was misconfigured
3. Brief CEO that the investigation is closed: false positive confirmed

IF CONFIRMED REAL:
1. Switch to full Security Incident procedure immediately
2. All steps from that procedure apply from the point of confirmation

RULE: The cost of treating a false positive seriously is low.
      The cost of treating a real incident as a false positive is catastrophic.

DOCUMENTATION REQUIRED:
- Security incident log entry regardless of outcome
- Investigation notes and conclusion
- Alert tuning record if threshold was adjusted\
"""
    },
    {
        "input": "Client is very unhappy with delivery quality AND hasn't paid the last invoice.",
        "output": """\
ISSUE TYPE: Client Escalation [with QA Failure + Invoice Overdue]
SEVERITY: High

Handle the relationship first — it determines how to approach both the quality and payment issues.

IMMEDIATE — RELATIONSHIP:
1. Account Director contacts client same day: listen first, do not defend
2. Separate the two issues internally — do not mix quality discussion with payment discussion
3. Understand: is the client withholding payment because of quality, or are these independent?

QUALITY ISSUE:
1. PM reviews all rejected items using the QA Failure procedure
2. Genuine bugs → fix immediately with no debate
3. Agreed fix timeline communicated to client in writing

PAYMENT:
1. If client is withholding due to quality → payment discussion pauses until quality is resolved
2. If withholding is separate → Finance follows Invoice Overdue procedure in parallel
3. Do not send aggressive payment demands while the quality issue is unresolved — it will end the relationship

ESCALATION:
- Relationship at serious risk → CEO joins within 24 hours
- Legal threat received → Legal Officer immediately

DOCUMENTATION REQUIRED:
- Client communication log
- Quality fix timeline
- Invoice status record\
"""
    },
]

# ---------------------------------------------------------------------------
# GENERATOR
# ---------------------------------------------------------------------------

def build_examples():
    examples = []

    for key, proc in PROCEDURES.items():
        msgs = MESSAGES[key]
        for msg in msgs:
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": msg},
                    {"role": "assistant", "content": proc["response"]},
                ]
            })

    for edge in EDGE_CASES:
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": edge["input"]},
                {"role": "assistant", "content": edge["output"]},
            ]
        })

    return examples


def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    examples = build_examples()
    random.shuffle(examples)

    split = int(len(examples) * 0.85)
    train = examples[:split]
    validation = examples[split:]

    out_dir = os.path.dirname(os.path.abspath(__file__))
    write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(out_dir, "validation.jsonl"), validation)

    print(f"Total examples : {len(examples)}")
    print(f"Train          : {len(train)}")
    print(f"Validation     : {len(validation)}")
    print(f"Issue types    : {len(PROCEDURES)}")
    print(f"Edge cases     : {len(EDGE_CASES)}")
    print()
    print("Files written:")
    print(f"  {os.path.join(out_dir, 'train.jsonl')}")
    print(f"  {os.path.join(out_dir, 'validation.jsonl')}")


if __name__ == "__main__":
    main()
