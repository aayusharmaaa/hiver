# VirginTrains taxonomy calibration: labeling guide

You will label **200** customer support cases from VirginTrains' Twitter support account. Your labels are the human reference: they are compared with the system's candidate intents to see which intents are real, which are confused with each other, and which should be merged, split or renamed before the taxonomy is frozen.

The cases come from a held-back reserve. None of them is one of the 250 golden evaluation cases.

## 1. Ground rules

- **Decide from the conversation first.** In the labeling tool the system's suggestion stays hidden until you click *Reveal*. Label, then (if you want) reveal it and compare.
- **Do not follow the suggestion blindly.** It comes from clustering opening messages, it is sometimes plainly wrong (see the worked examples), and the point of this exercise is to find where it is wrong. High agreement is not the goal; honest labels are.
- **Use only what is visible** in the public conversation. Do not guess about DMs, phone calls, or later events.
- **Do not use model output** (ChatGPT, Gemini or any other) to fill labels. This phase must be human-labelled.
- **Never leave a half-labelled case.** The four required fields are `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`; `human_notes` is optional except where this guide says otherwise.
- When you are unsure, give your best label and write a note. A confident wrong label is worse than a flagged uncertain one.

## 2. How to run the labeling

```
python scripts/label_taxonomy_calibration.py
```

- One case at a time, with progress `X / 200`, previous / next, and a jump list.
- Each *Save* checkpoints to `data/processed/taxonomy_calibration.csv` immediately. Close the tool whenever you like; start it again and it opens at your first unlabelled case.
- The tool writes only the five `human_*` columns. Case ids, tweet ids and sampling metadata cannot be changed from it, and it refuses to start if they were changed on disk.
- `data/processed/taxonomy_calibration_label_audit.jsonl` records every save (time, before/after, whether you had revealed the suggestion).
- Alternative without the tool: fill the five `human_*` columns directly in the CSV (matched by `case_id`) using the same vocabulary. Do not edit any other column.

## 3. What counts as the primary customer intent

The **primary intent is the customer's main request or complaint as the conversation shows it**, not the first words, not the agent's reply, and not the topic of the disruption that provoked it.

1. Is there a question or request the agent can act on? If yes, pick the intent for the **topic of that request** (status, ticket, refund claim, seat, Wi-Fi, catering...).
2. If there is no request, is the customer venting about a journey or service? Pick the complaint intent that matches what they are unhappy about.
3. If there is no request and no complaint, is it thanks or a compliment? `praise_positive_feedback`.
4. If it is banter, a joke, a photo, or general chatter that asks for nothing: `chitchat_non_support`.
5. Only if you still cannot say what the message is for: `unclear_or_media_only`.

Multi-intent cases:

- Choose the request the **agent would have to act on first**, and add `also: <intent_name>` to `human_notes`.
- A complaint plus an explicit question ("how do I claim a refund?"): the explicit question usually decides.
- Sarcasm is read as what it means ("thanks for 0 minutes to change trains" is a complaint).
- If the opening line is not the real request, label the real request and note `real request: ...`.
- Later turns count: the customer often clarifies what they wanted.
- Disruption is context, not an intent by itself: someone stranded by a delay who asks whether their ticket is valid asks a ticket question.

## 4. The intents

### `service_status_delay_enquiry`

Customer asks whether a service is running, delayed, or cancelled, or asks for an update on a delay they are currently experiencing. Merged from three clusters whose customer goal and agent replies are indistinguishable in the data.

- **Fits when:** Asks whether a named train or service is running, delayed, or cancelled, or when it will depart or arrive. Asks for news or an update while currently delayed, waiting, or stuck. Asks how announced disruption (engineering works, flooding, strikes) affects travel today.
- **Does not fit when:** Asks about Delay Repay, a refund, or compensation (use delay_repay_refund_claim). Reports or vents about a disrupted journey with no question (use journey_disruption_complaint). Asks whether a ticket is valid on a different train (use ticket_booking_query).

### `journey_disruption_complaint`

Venting or reporting a disrupted or overcrowded journey without a clear question; broad and heterogeneous (lost property and accessibility requests hide here).

- **Fits when:** Reports or complains about a cancelled, late, crowded, or short-formed train without asking a specific question. Complains about announcements or information given during the disruption.
- **Does not fit when:** Contains a concrete question about a service's status (use service_status_delay_enquiry). Complaint is mainly about staff behaviour or an unanswered complaint (use customer_service_complaint). Complaint is mainly about a missing or unhonoured seat reservation (use seat_reservation_issue).

### `chitchat_non_support`

Greetings, jokes, photos, and general chatter that do not request support.

- **Fits when:** Greeting, joke, photo, or remark that asks for nothing and complains about nothing.
- **Does not fit when:** Any complaint, question, or safety remark, however casual (use the matching support intent). Thanks or compliments aimed at the journey or staff (use praise_positive_feedback).

### `praise_positive_feedback`

Thanks, compliments for staff or journeys, and positive feedback.

- **Fits when:** Thanks, compliments for staff or a journey, or positive feedback with no request.
- **Does not fit when:** Sarcastic thanks that is really a complaint (use the complaint intent). Thanks sent as a reply that confirms a resolution (keep the intent of the original request).

### `ticket_booking_query`

Questions about ticket validity, advance tickets, bookings, using a ticket on another train, or app/booking problems.

- **Fits when:** Asks about ticket validity, advance or off-peak rules, using a ticket on another train, tickets or collection, or app or booking problems.
- **Does not fit when:** Asks for delay compensation (use delay_repay_refund_claim). Asks whether a train is running (use service_status_delay_enquiry). Asks about seat reservations only (use seat_reservation_issue).

### `seat_reservation_issue`

Reserved seats missing, double-booked, or not honoured; overcrowding and people standing.

- **Fits when:** Reserved seat missing, double-booked, unavailable, or not honoured; asks where unreserved seats are; seat reservation fees or coach choice.
- **Does not fit when:** General overcrowding complaint with no reservation element (use journey_disruption_complaint).

### `customer_service_complaint`

Complaint about staff, customer service, unanswered complaints or phone lines, or the quality of an earlier brand response.

- **Fits when:** Complains about staff conduct, an unhelpful or rude response, an unanswered complaint, or phone or contact lines. Replies to an earlier brand answer to say it was inadequate.
- **Does not fit when:** Complaint about the disruption itself with no customer-service element (use journey_disruption_complaint). Asks for compensation or a refund as the main request (use delay_repay_refund_claim).

### `delay_repay_refund_claim`

Customer asks about, chases, or disputes a Delay Repay claim, refund, or compensation for a delayed or cancelled journey.

- **Fits when:** Asks how to claim, chases, or disputes Delay Repay, a refund, or compensation for a delayed or cancelled journey.
- **Does not fit when:** Asks for a ticket-price refund because of a change of plans or a ticketing error (use ticket_booking_query). Asks for compensation for the wifi, catering, or seats specifically (use that intent; note the refund request in human_notes).

### `first_class_catering_issue`

First class catering, lounge, upgrade, or reduced first class service.

- **Fits when:** First class catering, lounge access, upgrades, or reduced first-class service.
- **Does not fit when:** Standard class food or trolley remarks (use journey_disruption_complaint or chitchat_non_support, as fits).

### `onboard_wifi_issue`

Onboard Wi-Fi not connecting, not working after paying, or paid-for access disputes.

- **Fits when:** Onboard wifi not connecting, slow, not working after payment, or portal or code problems.
- **Does not fit when:** Wifi mentioned only as a passing joke or sarcasm in an otherwise different complaint (use that complaint's intent; note in human_notes).

### `unclear_or_media_only`  (fallback)

Opening message has too little text after cleaning (mention only, photo/link only, fewer than 3 content words).

- **Fits when:** Opening message has no usable text (mention only, photo or link only, a single word) and the request cannot be inferred.
- **Does not fit when:** Short but clear requests ("any update?", "refund?") (use the matching intent).

## 5. Choosing between confusable intents

Notes for the pairs the system confuses most:

- `service_status_delay_enquiry` vs `journey_disruption_complaint`: A question about a service's status or timing is a status enquiry; venting or reporting disruption with no question is a complaint. If both are present, choose the main request and note the other.
- `chitchat_non_support` vs `praise_positive_feedback`: Praise is directed at the journey or staff. Chit-chat asks for and praises nothing. Doubtful warmth is chit-chat.
- `chitchat_non_support` vs `customer_service_complaint`: Casual or sarcastic wording that criticises staff or a reply is a complaint, not chit-chat.
- `chitchat_non_support` vs `journey_disruption_complaint`: Any complaint about the journey is a disruption complaint, even if short or jokey.
- `delay_repay_refund_claim` vs `ticket_booking_query`: Delay-related compensation is a Delay Repay claim; refunds or changes caused by the customer's own ticket or plans are ticket queries.
- `delay_repay_refund_claim` vs `customer_service_complaint`: If the main ask is money for a delay, it is a claim. If the main grievance is how the brand treated the customer, it is a service complaint.
- `seat_reservation_issue` vs `journey_disruption_complaint`: Mentions a reservation or a specific seat is a seat issue; general crowding or standing is a disruption complaint.
- `ticket_booking_query` vs `service_status_delay_enquiry`: "Can I use my ticket on the next train?" during disruption is a ticket query; "is the next train running?" is a status enquiry.
- `customer_service_complaint` vs `journey_disruption_complaint`: Staff, response, or contact failures are service complaints; the train itself being late, cancelled, or crowded is a disruption complaint.

General tie-breaks:

- Label the customer's **main request**. If the conversation shows the real request is different from the opening line, label the real request and say so in `human_notes`.
- If two intents both fit, choose the one the agent would have to act on first, and name the other in `human_notes` as `also: <intent>`.
- Use `unclear_or_media_only` only when you cannot tell what the customer wants even after reading the whole conversation. Short but clear requests ("any update?", "refund?") belong to their normal intent.
- Do not choose an intent because it is the system's suggestion. Decide first, then compare.
- If none of the intents fit and you would write a new one, put `NEW:<snake_case_name>` in `human_intent` and describe it in `human_notes`.

## 6. When to use the fallback `unclear_or_media_only`

- Use it when, **after reading the whole conversation**, you cannot tell what the customer wants: a bare mention, a photo or link with no words, or text too short to interpret.
- Do **not** use it for a message that is clear but awkward to classify, for a short but clear request ("any update?"), or for a non-support message (that is `chitchat_non_support`).
- Label what you can see; do not guess what a link or image might contain.

## 7. When to use `NEW:<name>`

- Only when **no existing intent fits and you would expect the same kind of case to recur**. Use `snake_case`, e.g. `NEW:lost_property`.
- A note is **required**: describe the new intent in one line (`new intent: <name>: <definition>`).
- For a one-off, pick the closest existing intent and explain in `human_notes`.
- A `NEW:` label is a proposal. Whether it becomes an intent is decided later by the reviewer, after the comparison report.

## 8. `human_resolution_type`: what the agent did

Pick the agent's main move from the agent's words. Do not copy the system's `auto_resolution_type`.

- `refund`: the agent discusses a refund (including refusing one)
- `compensation`: the agent discusses compensation or a claim, e.g. Delay Repay
- `account_action`: the agent says it has changed, or will change, a booking or account
- `redirected_to_dm`: the agent asks the customer to continue in direct messages
- `redirected_to_other_operator`: the agent hands the customer to another train operator
- `escalated`: the agent refers the customer to Customer Relations, a formal complaint route, or forwards it internally
- `troubleshooting`: the agent gives steps to try
- `self_service`: the agent points to a link, the app, the website, or live updates
- `feedback_acknowledged`: the agent thanks the customer or says feedback will be passed on
- `information_provided`: the agent answers with information that does not fit a more specific type above
- `clarification_requested`: the agent only asks a question
- `other`: the agent replied, but nothing above fits
- `unresolved`: the brand never replied, or the customer was left hanging
- `unclear`: you cannot tell

When two types could apply, take the **most specific** one: `refund`, `compensation`, `account_action`, `escalated`, and the redirects come before `troubleshooting`; `troubleshooting` before `self_service`; `self_service` before `feedback_acknowledged`; `information_provided` and `clarification_requested` are the general cases; `other` is the last resort.

## 9. `human_resolved`

`yes` / `no` / `unclear`.

- **yes**: the request was visibly met or closed in the public conversation (the customer confirms, or the agent's reply fully answers it and nothing is left open).
- **no**: something is visibly still open, the customer pushes back, or the brand never replied.
- **unclear**: you cannot tell from what is public, for example the thread moves to DMs.

A polite "thanks" after a reply that did not answer the question is not a resolution.

## 10. `human_escalation_signal`

Describes a signal visible in the conversation, not a policy about what to do. Choose the strongest that applies; mention others in `human_notes`.

- `none`: no sign that the case needs more than a normal reply
- `formal_complaint_or_customer_relations`: the customer wants to complain formally, or the agent points to Customer Relations / a complaints form, or the case is already with them
- `safety_or_vulnerability`: physical safety, harassment or aggression, a stranded or distressed customer, or someone who says they are vulnerable
- `accessibility_or_assistance`: the customer needs, or reports a failure of, assisted travel, disability access, or help boarding
- `legal_media_or_ombudsman_threat`: the customer threatens legal action, the press, an ombudsman or a regulator
- `repeated_unresolved_contact`: the customer says they already contacted the brand (more than once, or a long time ago) without a useful answer
- `needs_account_or_booking_lookup`: the agent cannot help without the customer's booking, account, or claim details
- `other`: none of the above fits; explain in `human_notes`

## 11. `human_notes`

Free text. Write a note whenever the case is ambiguous, two intents fit, the opening line is misleading, or you used `NEW:`. These notes feed the disagreement report. Suggested prefixes:

- `also: <intent_name>`: a second intent also fits. The comparison tool reads this, so write the exact intent name, one `also:` per intent.
- `real request: ...`: the opening line is not the real request.
- `unsure: ...`: a genuine coin-flip between two labels (name both).
- `resolution: ...`: your resolution type or resolved label differs from what the system would guess, and why.
- `new intent: <name>: <one-line definition>`: when you used `NEW:<name>`.

## Worked examples (from the train split)

Every case below is a **train** case, never one of the cases you are labelling and never a golden case. The quotes are copied from the dataset. Each reading shows how the rules apply and is one defensible reading, not ground truth. The system's suggestion is printed so you can see how often it disagrees with a sensible reading.

### The system's suggestion is sometimes wrong

These train cases show the suggestion pointing at an intent the conversation does not support. The point is not that the suggestion is bad, only that you must decide from the conversation.

**`case_1185`**  
> **CUSTOMER:** @VirginTrains Hi- does the new app have an option to buy an open return?  
> **AGENT:** @115921 Yes it does. You need to select the return journey then in the ticket flexibility menu you can select open return ^CB  
> **CUSTOMER:** @VirginTrains Thanks, see it now. Real shame about how poor the new App is though.  

- System suggestion (not authoritative): `delay_repay_refund_claim`
- Reading: The customer asks whether the new app can sell an open return: a question about ticket types in the app. Nothing in the conversation concerns a delay or a refund. The closest existing intent is `ticket_booking_query`.

**`case_2232529`**  
> **CUSTOMER:** @VirginTrains Hi - for veterans, is it there vererans badge and old id sufficient evidence?  
> **AGENT:** @146476 Do they receive ID when they they leave the forces? ^PA  
> **CUSTOMER:** @VirginTrains No they get a veterans badge, and have their old id - my partner has his old united nations id.  
> **AGENT:** @146476 That will be fine Lucy. ^PA  
> **CUSTOMER:** @VirginTrains Perfect! Thank you!!  

- System suggestion (not authoritative): `customer_service_complaint`
- Reading: A question about whether a veterans badge and old ID are enough evidence; the agent confirms. There is no complaint. The closest existing intent is `ticket_booking_query` (fare or discount eligibility). If you feel fare-eligibility questions deserve their own intent, say so in `human_notes`.

**`case_1707839`**  
> **CUSTOMER:** I’m honestly not on a retainer from @VirginTrains but by heck I love this company! Customer focussed, great quality and lovely staff. 👌👍  
> **AGENT:** @263674 Wonderful to hear, Amanda 😊 Thank you for your feedback! We'll be sure to pass your comments on ^HP  

- System suggestion (not authoritative): `chitchat_non_support`
- Reading: Unprompted praise for the company and its staff; the agent thanks the customer and passes it on. This is `praise_positive_feedback`.

**`case_1205564`**  
> **CUSTOMER:** Really glad I forked out 5 quid for the @VirginTrains onboard internet. If this tweet posts it will be a miracle. 😂 https://t.co/U16pfyIRUg  
> **AGENT:** @403156 Try the support team on 0330 088 1271 for assistance, Clare ^MW  

- System suggestion (not authoritative): `customer_service_complaint`
- Reading: The customer says they are "really glad" they paid for the onboard internet and that posting the tweet would be a miracle. The sarcasm is the complaint, and it is about the Wi-Fi: `onboard_wifi_issue`. The agent gives a support number.

### More than one intent in a case

Choose the request the agent has to act on first and name the other in `human_notes` as `also: <intent>`.

**`case_1139723`**  
> **CUSTOMER:** @VirginTrains 10:50 from New Street to Euston is an absolute disgrace. You cannot move down the isle. Everybody shoulder to shoulder. Fire safety comes to mind. Let’s hope there isn’t an incident. Train manager is great and admits this is horrendous. How do I claim a refund?  
> **AGENT:** @388284 If you're unhappy with your experience onboard this morning please get in touch through our website: https://t.co/t20UbyOCSn ^CB  

- System suggestion (not authoritative): `journey_disruption_complaint`
- Reading: Two things: a complaint about an overcrowded train and the explicit question "How do I claim a refund?". The question is what the agent has to act on, so a defensible primary intent is `delay_repay_refund_claim` with `also: journey_disruption_complaint`. The agent's reply points to a general complaints page and does not answer the refund question, which matters for `human_resolved`.

**`case_1353023`**  
> **CUSTOMER:** @VirginTrains I have a train Euston to Manchester@8:40. Is this service still running?  
> **AGENT:** @435549 This service is running, but may meet with delay at EUS due to awaiting train crew. ^BT  
> **CUSTOMER:** @VirginTrains Would my ticket be valid for travel tomorrow?  
> **AGENT:** @435549 Hi Dav, it will be yes ^HP  
> **CUSTOMER:** @VirginTrains Thanks @VirginTrains!!!  

- System suggestion (not authoritative): `service_status_delay_enquiry`
- Reading: The first message asks whether a train is running, so the main request is `service_status_delay_enquiry`. Later the customer also asks whether the ticket is valid tomorrow: add `also: ticket_booking_query`.

**`case_2669898`**  
> **CUSTOMER:** Thanks a lot @VirginTrains! Mislead advise has meant I’ve checked out of my hotel room early, but now I’m stuck at Euston for 4 hours! Disgusting! Apparently Advance Off-Peak is not the same as an Advance ticket during Off-Peak! @306277 @418 #TrainYourStaff  
> **AGENT:** @751919 That is correct, an Advance is only valid on one train only, James ^MW  
> **CUSTOMER:** @VirginTrains We have checked out of our hotel room based on misleading information and the team leader at the ticket office’s advice was a sarcastic ‘Good Luck’ I cannot believe how poor your service is! @28792 @418 https://t.co/HP0q56jeIX  
> **CUSTOMER:** @VirginTrains @28792 @418 I am logging a formal complaint. Thanks to Adrian at Euston for being the most rude and obnoxious person I have ever spoken to!@306277 @418  
> **AGENT:** @751919 Were you looking to make a complaint in regards to this? ^MW  

- System suggestion (not authoritative): `ticket_booking_query`
- Reading: Anger about misleading advice and a long wait, a specific ticket-validity point (Advance Off-Peak versus Advance), and finally "I am logging a formal complaint" about a staff member's behaviour. The conversation ends on the complaint, so `customer_service_complaint` with `also: ticket_booking_query` is a defensible reading, and the escalation signal `formal_complaint_or_customer_relations` is visible. `ticket_booking_query` as primary is also defensible; the skill being shown is the note: pick the one you judge the customer most wants addressed and name the other.

### When the opening line is not the real request

Label the real request and say that you did in `human_notes`.

**`case_1620103`**  
> **CUSTOMER:** @VirginTrains had an advance 1st Lon-Manc for 1517. If I stay at Euston can I use for train when they restart?  
> **AGENT:** @496556 You can travel tomorrow, as close to your booked time as possible ^MM  
> **CUSTOMER:** @VirginTrains Sorry - What I’m asking is can I still use later today  
> **AGENT:** @496556 You can travel later today, Matt. Live Updates can be found via https://t.co/G0V1vybTwA ^MM  
> **CUSTOMER:** @VirginTrains Thanks. The live updates not v helpful FYI. They said 1635 was on time until 20 mins so we stayed and now not happening.  

- System suggestion (not authoritative): `service_status_delay_enquiry`
- Reading: The opening asks whether an advance ticket can be used when trains restart, but the customer then restates the question: "What I'm asking is can I still use later today". The real request is ticket validity (`ticket_booking_query`), even though the context is a disruption.

### Confusable intents

Short cases for the pairs the system confuses most. Each one shows the deciding detail.

**`case_1451163`**  
> **CUSTOMER:** Thanks to @VirginTrains I have 0 minutes to get from one train to another. 🤔🙃👏🏽  
> **AGENT:** @456756 Which service are you on Ben? ^CB  

- System suggestion (not authoritative): `journey_disruption_complaint`
- Reading: Status versus complaint. A sarcastic complaint with no question. With nothing asked about status or timing, this fits `journey_disruption_complaint` better than a status enquiry.

**`case_1330426`**  
> **CUSTOMER:** @VirginTrains any chance that the 13:43 from Stockport to London Euston will still be running?  
> **AGENT:** @430383 It's unlikely at this stage I'm afraid Oli ^CB  
> **CUSTOMER:** @VirginTrains I'm having to go to Sheffield to get to London now. I assume I can get my Virgin ticket refunded?  
> **AGENT:** @430383 If you're delayed by over 30 minutes please claim for this here: https://t.co/t0vCSebzQk ^CB  
> **CUSTOMER:** @VirginTrains Cheers, much appreciated  

- System suggestion (not authoritative): `service_status_delay_enquiry`
- Reading: Status versus claim. "Any chance that the 13:43 ... will still be running?" is a direct status question (`service_status_delay_enquiry`). The later refund question is a second request: note `also: delay_repay_refund_claim`.

**`case_1694230`**  
> **CUSTOMER:** @VirginTrains I want money back for two tickets - Manchester to Euston yesterday. Excruciating delayed journey. Please advise.  
> **AGENT:** @514100 Hi Nicola, you can claim compensation via our Delay Repay form here: https://t.co/MowTo2slfk ^HP  

- System suggestion (not authoritative): `delay_repay_refund_claim`
- Reading: Claim versus ticket. "I want money back for two tickets ... delayed journey": the money is claimed because of the delay, so `delay_repay_refund_claim`.

**`case_1329607`**  
> **CUSTOMER:** @VirginTrains Do we get our ticket fare back???? I’m having to stay an extra night in London can’t even move in Euston let alone get a train if there was to be one!!!  

- System suggestion (not authoritative): `service_status_delay_enquiry`
- Reading: Claim versus status. "Do we get our ticket fare back????" while stranded, and no reply. A refund question tied to a disruption is `delay_repay_refund_claim`; the stranded detail is context (and worth a note for the escalation signal).

**`case_494597`**  
> **CUSTOMER:** This rather busy and delayed @virgintrains trip would be fine if the wife would stop moaning #doghouse  
> **AGENT:** @232950 We would suggest you tell your wife stop whinging and fetch you some beer ^MM  
> **CUSTOMER:** @VirginTrains I’m as likely to get a beer from her as you are to make the Christmas drinks 😂  
> **AGENT:** @232950 And you know this ^CB 😂 https://t.co/Qg08uQQZsZ  

- System suggestion (not authoritative): `chitchat_non_support`
- Reading: Chitchat versus praise. A joke about the customer's wife on a busy, delayed train; the agent jokes back. There is no request, no real praise and no real complaint: `chitchat_non_support`.

**`case_875344`**  
> **CUSTOMER:** @VirginTrains thank you to the guard who helped my daughter when she was penned in by a woman yelling at another. #knightinshiningarmour  
> **AGENT:** @327829 Glad to hear that, which service were you on exactly? ^MW  
> **CUSTOMER:** @VirginTrains She was coming in on the 4:15 Norwich train to LPY. It’s a nightmare everyone so grumpy but your guard saved her  

- System suggestion (not authoritative): `chitchat_non_support`
- Reading: Chitchat versus praise. "Thank you to the guard who helped my daughter": thanks aimed at a staff member, which the agent follows up so it can be passed on. This is `praise_positive_feedback`.

**`case_1148900`**  
> **CUSTOMER:** Did not just expect the announcement on my @VirginTrains to be about a surfboard not being secured enough 😂  
> **AGENT:** @390371 They're the best announcements 😂 ^MW  

- System suggestion (not authoritative): `customer_service_complaint`
- Reading: Chitchat versus complaint. A joke about an announcement; the agent replies in kind. Humour about staff is not a complaint about staff: `chitchat_non_support`.

**`case_2694730`**  
> **CUSTOMER:** What is the point in reserving a seat if @VirginTrains put me in the unreserved coach anyway? Absolute shambles.  
> **AGENT:** @757313 Hi there, which service was this for? ^HP  
> **CUSTOMER:** @VirginTrains 17.40 London Euston to Glasgow central  
> **AGENT:** @757313 Sorry to hear this, we weren't aware of the issue. Have you been able to speak to the onboard team about this? ^HP  
> **CUSTOMER:** @VirginTrains There's no staff about and basically no one can even tell what each coach is anyways. They announced over speaker that J is unreserved and that's where my ticket is booked for?  

- System suggestion (not authoritative): `seat_reservation_issue`
- Reading: Seat versus journey. "What is the point in reserving a seat if [they] put me in the unreserved coach anyway?" The reservation was not honoured: `seat_reservation_issue`.

**`case_813058`**  
> **CUSTOMER:** #HotStuff ?!?!?! Maybe in the cup, but not the carriage. En route to #Manchester with the air con rammed on cold. Oh @VirginTrains 🙈 https://t.co/XB5zql7DLg  
> **AGENT:** @313741 Oh no, if you speak to staff, they can adjust the temp for you. ^PA  
> **CUSTOMER:** @VirginTrains Train Manager said he’d done it an hour ago! #Frostbite  
> **AGENT:** @313741 They would alter this as best they can ^CB  

- System suggestion (not authoritative): `journey_disruption_complaint`
- Reading: Seat versus journey. The air conditioning is too cold and staff say they adjusted it. There is no delay, seat or ticket issue, so no intent fits well. The closest is `journey_disruption_complaint` (onboard experience); write a note. If you find several cases like this, that is how a possible new intent gets noticed.

**`case_590467`**  
> **CUSTOMER:** @VirginTrains hi just wondered if my train ticket will be valid to travel tomorrow instead of today due to overcrowding? Really don't fancy standing for 3 hours from oxenholme to London  
> **AGENT:** @259652 Yes Liv. Tickets for travel today will be valid tomorrow. ^BT  

- System suggestion (not authoritative): `ticket_booking_query`
- Reading: Ticket versus status. "Will my ticket be valid to travel tomorrow instead of today due to overcrowding?" The request is ticket validity (`ticket_booking_query`); the overcrowding is only the reason.

**`case_1329593`**  
> **CUSTOMER:** @VirginTrains What if you have an off peak return and are trying to travel tonight?  

- System suggestion (not authoritative): `ticket_booking_query`
- Reading: Ticket versus status. "What if you have an off peak return and are trying to travel tonight?" is a short, clear ticket question (`ticket_booking_query`). Short is not the same as unclear.

### Using the fallback

`unclear_or_media_only` is for messages whose purpose you cannot tell, not for messages you find hard to classify.

**`case_1162203`**  
> **CUSTOMER:** @VirginTrains https://t.co/iFYhyCRqar  
> **AGENT:** @651413 Alright ah kid. ^PA  

- System suggestion (not authoritative): `unclear_or_media_only`
- Reading: Only a mention and a link, and the agent's reply does not reveal a request either. Nothing visible says what the customer wants: `unclear_or_media_only`. Label what you can see, not what the link might contain. (Check the whole conversation first: a bare link that is followed by clear chat or a clear question is labelled by that chat or question.)

**`case_1163025`**  
> **CUSTOMER:** @120576 @VirginTrains @393592 DBS class 67?  

- System suggestion (not authoritative): `unclear_or_media_only`
- Reading: A one-line fragment of someone else's thread ("DBS class 67?") with no context and no reply. You cannot tell whether it is a question, a joke or something else: `unclear_or_media_only`.

**`case_1527238`**  
> **CUSTOMER:** Our nights team have just arrived into the briefing room getting ready to fight crime and keep you safe  

- System suggestion (not authoritative): `praise_positive_feedback`
- Reading: An organisation-style broadcast post, not a request. It is clear what the message is, so it is not unclear. `chitchat_non_support` is the closer fit; add a note such as "not a customer request".

### When to propose NEW:<name>

Use `NEW:<snake_case_name>` only when no existing intent fits and you would expect the same kind of case to recur. Describe the intent in `human_notes`. For a one-off, pick the closest intent and explain. The registry already notes that lost-property and accessibility requests currently hide inside broader intents. Whether to keep any new intent is decided later by the reviewer, after the comparison report.

**`case_2507047`**  
> **CUSTOMER:** @VirginTrains hope you can held I've left a plastic bag above seat D27 from Euston to Glasgow I got off at Warrington it is a House of Lords bag can you locate it for me and drop it off at Bank Quay  
> **AGENT:** @714800 If you leave a message with lost property on 03331 031 031 option 1, 3 they will call back if it's found ^MW  
> **CUSTOMER:** @VirginTrains I've contacted the number got a no for Glasgow lost property but it's not a good number can you help  
> **AGENT:** @714800 The only other thing we can suggest would be filling in this form, Neil - https://t.co/RzKNppWsnJ ^MW  

- System suggestion (not authoritative): `praise_positive_feedback`
- Reading: The customer left a bag on the train and asks the brand to locate it; the agent gives a lost-property number and a form. If you meet several cases like this, `NEW:lost_property` is a reasonable proposal.

**`case_1500244`**  
> **CUSTOMER:** @VirginTrains disability assistance sent to wrong carriage both ends of journey, left stranded while they went to help others-why do I book?  
> **AGENT:** @468308 Oh no, have you been assisted now, Lj? Sorry to hear about this by the way ^MW  
> **CUSTOMER:** @VirginTrains Ironically by passenger was abandoned for,injured in process system needs improving for disabled,staff shouldn't exit train b4 disabled off  
> **AGENT:** @468308 I see, sorry to hear this, were you looking to make a complaint in regards to this? ^MW  
> **CUSTOMER:** @VirginTrains Been complaining for yrs just want better services for disabled.I hear rumours of training &amp; policy clarity but many staff still lacking  

- System suggestion (not authoritative): `seat_reservation_issue`
- Reading: Disability assistance was sent to the wrong carriage and the customer says the system needs improving for disabled passengers. Existing intents fit poorly. Label the closest one and set the escalation signal to `accessibility_or_assistance`. Use `NEW:accessibility_assistance` only if you think such cases deserve their own intent.

### Resolution type: what the agent did

Judge the agent's main move, using the most specific type that fits. A link inside a compensation or refund reply does not make it `self_service`.

**`case_1694230`**  
> **CUSTOMER:** @VirginTrains I want money back for two tickets - Manchester to Euston yesterday. Excruciating delayed journey. Please advise.  
> **AGENT:** @514100 Hi Nicola, you can claim compensation via our Delay Repay form here: https://t.co/MowTo2slfk ^HP  

- System suggestion (not authoritative): `delay_repay_refund_claim`
- Reading: "You can claim compensation via our Delay Repay form": `compensation`. A link is present, but a compensation claim is more specific than `self_service`.

**`case_1295272`**  
> **CUSTOMER:** @VirginTrains service to and from NEC to London today unacceptable. How does one obtain refund?  
> **AGENT:** @422794 You can apply for this here https://t.co/oryZQKeenc ^BT  

- System suggestion (not authoritative): `delay_repay_refund_claim`
- Reading: "You can apply for this here <link>" in answer to a refund question: `refund`.

**`case_1235021`**  
> **CUSTOMER:** @VirginTrains do you know if the 9pm or 9.40pm from Euston to Manchester will be running normal later on?  
> **AGENT:** @409675 1/2 Hi Kerrim, this service is booked to run but we are expecting residual delays. You can keep up to date with our services  
> **AGENT:** @409675 2/2 via our Live Updates link here: https://t.co/kn4iubBMDU ^HP  
> **CUSTOMER:** @VirginTrains Okay thanks  

- System suggestion (not authoritative): `service_status_delay_enquiry`
- Reading: The agent says the service is booked to run with residual delays and points to the Live Updates link: `self_service` (the pointer is the main content). If the answer itself were in the reply and the link were incidental, use `information_provided`.

**`case_2271509`**  
> **CUSTOMER:** @VirginTrains crazy full 10.35am York to Southampton! 😡 standing room only, ticket/seat issues, packed btwn coaches what is going on?  
> **AGENT:** @660861 No our service I'm afraid, you would need to speak with @121428 regarding this. ^BT  

- System suggestion (not authoritative): `seat_reservation_issue`
- Reading: "No our service I'm afraid, you would need to speak with [another operator]": `redirected_to_other_operator`.

**`case_2392332`**  
> **CUSTOMER:** @VirginTrains Hi. I haven't heard anything yet re my refund due to delay.  
> **AGENT:** @688949 Do you have a VT reference number? ^PA  
> **CUSTOMER:** @VirginTrains VT-261017-3514  
> **AGENT:** @688949 Can you please DM us so we can discuss this please. ^PA https://t.co/hhfk9c3ylv  

- System suggestion (not authoritative): `delay_repay_refund_claim`
- Reading: The agent asks for a reference, then "Can you please DM us": `redirected_to_dm`.

**`case_1451163`**  
> **CUSTOMER:** Thanks to @VirginTrains I have 0 minutes to get from one train to another. 🤔🙃👏🏽  
> **AGENT:** @456756 Which service are you on Ben? ^CB  

- System suggestion (not authoritative): `journey_disruption_complaint`
- Reading: The agent only asks "Which service are you on?": `clarification_requested`.

**`case_113998`**  
> **CUSTOMER:** @VirginTrains hi, I put in a delay repay claim on 6 Nov, had an email saying it had been received but not received any further communication (still not had a ref no.) Do I need to resubmit?  
> **AGENT:** @141322 I can see your case is with Customer Relations. It can take the team up to 28 days to issue a full response although they do aim to respond with 14 days were possible ^MM  
> **CUSTOMER:** @VirginTrains Great thanks, just wanted to check it hadn't slipped through the net somewhere :)  

- System suggestion (not authoritative): `delay_repay_refund_claim`
- Reading: "I can see your case is with Customer Relations": `escalated`.

**`case_1609645`**  
> **CUSTOMER:** @126035 @VirginTrains The Livery is to plain. https://t.co/A8iaYhpVF5  
> **AGENT:** @445319 @126035 Hi Mark. Sad to hear you're not a fan of our new makeover😔 But we will be sure to pass on your comments. ^BT  

- System suggestion (not authoritative): `chitchat_non_support`
- Reading: "We will be sure to pass on your comments": `feedback_acknowledged`.

**`case_1329593`**  
> **CUSTOMER:** @VirginTrains What if you have an off peak return and are trying to travel tonight?  

- System suggestion (not authoritative): `ticket_booking_query`
- Reading: The brand never replied: `unresolved`.

### Resolved: yes, no or unclear

Judge only what is visible in the public conversation.

**`case_1330426`**  
> **CUSTOMER:** @VirginTrains any chance that the 13:43 from Stockport to London Euston will still be running?  
> **AGENT:** @430383 It's unlikely at this stage I'm afraid Oli ^CB  
> **CUSTOMER:** @VirginTrains I'm having to go to Sheffield to get to London now. I assume I can get my Virgin ticket refunded?  
> **AGENT:** @430383 If you're delayed by over 30 minutes please claim for this here: https://t.co/t0vCSebzQk ^CB  
> **CUSTOMER:** @VirginTrains Cheers, much appreciated  

- System suggestion (not authoritative): `service_status_delay_enquiry`
- Reading: `yes`. The agent answers (the train is unlikely to run; claim link for a delay over 30 minutes) and the customer says "Cheers, much appreciated".

**`case_1620103`**  
> **CUSTOMER:** @VirginTrains had an advance 1st Lon-Manc for 1517. If I stay at Euston can I use for train when they restart?  
> **AGENT:** @496556 You can travel tomorrow, as close to your booked time as possible ^MM  
> **CUSTOMER:** @VirginTrains Sorry - What I’m asking is can I still use later today  
> **AGENT:** @496556 You can travel later today, Matt. Live Updates can be found via https://t.co/G0V1vybTwA ^MM  
> **CUSTOMER:** @VirginTrains Thanks. The live updates not v helpful FYI. They said 1635 was on time until 20 mins so we stayed and now not happening.  

- System suggestion (not authoritative): `service_status_delay_enquiry`
- Reading: `no`. The customer ends by saying the live updates were not helpful and the train they waited for is not happening. The customer pushes back.

**`case_75055`**  
> **CUSTOMER:** @VirginTrains Hi, can you chase up a refund from October for me please? Was supposed to be automatic, didn't happen, have sent in tickets, heard nothing.  
> **AGENT:** @132798 Hi, do you have a VT reference at all? ^LC  
> **CUSTOMER:** @VirginTrains Yep, is that OK to share here or do I need to go to DM? Thanks  
> **AGENT:** @132798 You can DM us this ^LC  
> **CUSTOMER:** @VirginTrains done thanks  

- System suggestion (not authoritative): `delay_repay_refund_claim`
- Reading: `unclear`. The agent asks for a reference, then tells the customer to DM it; the customer says "done thanks". The substance moves to DMs, so you cannot see whether the refund chase succeeded.

**`case_1329593`**  
> **CUSTOMER:** @VirginTrains What if you have an off peak return and are trying to travel tonight?  

- System suggestion (not authoritative): `ticket_booking_query`
- Reading: `no`. The request was never answered because the brand never replied.

### Escalation signal

This describes a signal in the conversation, not a policy. Choose the strongest that applies and mention the others in `human_notes`. No clear legal-threat or media-threat case turned up in the train cases reviewed for this guide, so there is no example for `legal_media_or_ombudsman_threat`; use it if you meet one.

**`case_1500244`**  
> **CUSTOMER:** @VirginTrains disability assistance sent to wrong carriage both ends of journey, left stranded while they went to help others-why do I book?  
> **AGENT:** @468308 Oh no, have you been assisted now, Lj? Sorry to hear about this by the way ^MW  
> **CUSTOMER:** @VirginTrains Ironically by passenger was abandoned for,injured in process system needs improving for disabled,staff shouldn't exit train b4 disabled off  
> **AGENT:** @468308 I see, sorry to hear this, were you looking to make a complaint in regards to this? ^MW  
> **CUSTOMER:** @VirginTrains Been complaining for yrs just want better services for disabled.I hear rumours of training &amp; policy clarity but many staff still lacking  

- System suggestion (not authoritative): `seat_reservation_issue`
- Reading: `accessibility_or_assistance`. The agent also asks whether the customer wants to make a complaint, so `formal_complaint_or_customer_relations` is visible too; mention it in the notes.

**`case_991875`**  
> **CUSTOMER:** Not impressed @VirginTrains explaining to someone who you have Aspergers, anxious and tearful asking for help, to then to be nearly put  
> **CUSTOMER:** @VirginTrains On the wrong train.. making me now x10 anxious, x10 more tearful and making this journey x100 more stressful then t needed to be  
> **AGENT:** @354951 Sorry to heat that Amy, where did this happen? ^PA  
> **CUSTOMER:** @VirginTrains York, I normally can do big journeys myself and it's a shame the one time I ask for it if backfires 😑  
> **AGENT:** @354951 Ah I see, this will be one for our colleagues at @120576 to help with. ^PA  

- System suggestion (not authoritative): `chitchat_non_support`
- Reading: `accessibility_or_assistance`: the customer says they have Aspergers, are anxious and tearful, and asked for help. `safety_or_vulnerability` is also arguable; say which you chose and why.

**`case_2771522`**  
> **CUSTOMER:** Nice one @VirginTrains for allowing five passengers to disturb everybody on the 11:40 to London from Glasgow. How many pro IRA songs do we need to hear in coach D before your staff ask them to be a little more considerate of EVERYBODY ELSE ON THE TRAIN?  
> **AGENT:** @774724 Have you spoken with the onboard team regarding this? ^CB  
> **CUSTOMER:** @VirginTrains They've walked passed it several times and said nothing. Nobody needs to tell them it's clear to see and hear!  
> **AGENT:** @774724 We'll certainly pass on your comments regarding this ^CB  
> **CUSTOMER:** @VirginTrains Thanks they're also drunk and super aggressive. It's very unpleasant and you passing on my comments isn't going to save this journey but thanks anyway 🙄  

- System suggestion (not authoritative): `journey_disruption_complaint`
- Reading: `safety_or_vulnerability`: drunk, aggressive passengers, and staff who walked past without acting.

**`case_2368116`**  
> **CUSTOMER:** @VirginTrains - Hate using Twitter for customer service, but been waiting 3 weeks for a refund due to massive delay and no one is responding via email. Help please!  
> **AGENT:** @683466 It can take the team up to 28 days to issue a full response although they do aim to respond with 14 days were possible ^MM  
> **CUSTOMER:** @VirginTrains I got a response within 1 day to say you needed more info to process the refund but nothing since giving you the extra info. Surely confirmation that you have everything you need isn’t too much to ask?  
> **AGENT:** @683466 The team will be looking into this, Ben ^MM  
> **CUSTOMER:** @VirginTrains Thanks. DM me if you need an email address to find the refund request.  

- System suggestion (not authoritative): `delay_repay_refund_claim`
- Reading: `repeated_unresolved_contact`: three weeks waiting for a refund and no reply to emails.

**`case_75055`**  
> **CUSTOMER:** @VirginTrains Hi, can you chase up a refund from October for me please? Was supposed to be automatic, didn't happen, have sent in tickets, heard nothing.  
> **AGENT:** @132798 Hi, do you have a VT reference at all? ^LC  
> **CUSTOMER:** @VirginTrains Yep, is that OK to share here or do I need to go to DM? Thanks  
> **AGENT:** @132798 You can DM us this ^LC  
> **CUSTOMER:** @VirginTrains done thanks  

- System suggestion (not authoritative): `delay_repay_refund_claim`
- Reading: `needs_account_or_booking_lookup`: the agent needs the customer's booking reference and moves to DM.

**`case_113998`**  
> **CUSTOMER:** @VirginTrains hi, I put in a delay repay claim on 6 Nov, had an email saying it had been received but not received any further communication (still not had a ref no.) Do I need to resubmit?  
> **AGENT:** @141322 I can see your case is with Customer Relations. It can take the team up to 28 days to issue a full response although they do aim to respond with 14 days were possible ^MM  
> **CUSTOMER:** @VirginTrains Great thanks, just wanted to check it hadn't slipped through the net somewhere :)  

- System suggestion (not authoritative): `delay_repay_refund_claim`
- Reading: `formal_complaint_or_customer_relations`: the agent confirms the case is already with Customer Relations.

**`case_1235021`**  
> **CUSTOMER:** @VirginTrains do you know if the 9pm or 9.40pm from Euston to Manchester will be running normal later on?  
> **AGENT:** @409675 1/2 Hi Kerrim, this service is booked to run but we are expecting residual delays. You can keep up to date with our services  
> **AGENT:** @409675 2/2 via our Live Updates link here: https://t.co/kn4iubBMDU ^HP  
> **CUSTOMER:** @VirginTrains Okay thanks  

- System suggestion (not authoritative): `service_status_delay_enquiry`
- Reading: `none`: a plain status question with a normal answer.

## 12. Before you finish

- All 200 cases show as labelled in the tool (the progress bar reads 200 / 200).
- Every `NEW:` label has a note.
- You did not change anything outside the `human_*` columns.
- Do not edit the decisions file yet: merge and split decisions are made *after* the comparison report, by the reviewer, in `configs/virgintrains_taxonomy_decisions.yaml`.

---
