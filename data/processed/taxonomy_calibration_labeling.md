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

### 001 · `case_1130049`

**First customer message:** You do not want to miss out on this GREAT opportunity w/ @VirginTrains ! Lead Talent Partner based in York! #HRJobs https://t.co/gmorX41mOd

**Conversation**

1. **CUSTOMER** `1130049` 2017-10-24 11:33: You do not want to miss out on this GREAT opportunity w/ @VirginTrains ! Lead Talent Partner based in York! #HRJobs https://t.co/gmorX41mOd
2. **AGENT** `1130048` 2017-10-24 11:34: @386337 With @120576 that would be 😃 ^MW

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `praise_positive_feedback` (cluster 8 `praise_positive_feedback`; runner-up `chitchat_non_support`, margin 0.0052)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 1130048, information_provided] "With @120576 that would be 😃"

Fill in CSV row `case_1130049`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 002 · `case_1387840`

**First customer message:** @VirginTrains can't fault the service these days....frigging sauna in carriage A en route to London Euston! #airconrequired #dontfart

**Conversation**

1. **CUSTOMER** `1387840` 2017-10-28 10:04: @VirginTrains can't fault the service these days....frigging sauna in carriage A en route to London Euston! #airconrequired #dontfart
2. **AGENT** `1387839` 2017-10-28 10:08: @442911 If you speak to the onboard team they'd look to alter this for you ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `praise_positive_feedback`, margin 0.0187)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 1387839, information_provided] "If you speak to the onboard team they'd look to alter this for you"

Fill in CSV row `case_1387840`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 003 · `case_2469339`

**First customer message:** @VirginTrains I submitted a delay repay claim for my journey in 5 Nov and received your automated email on 6 Nov, when can I expect to hear from you?

**Conversation**

1. **CUSTOMER** `2469339` 2017-11-15 10:05: @VirginTrains I submitted a delay repay claim for my journey in 5 Nov and received your automated email on 6 Nov, when can I expect to hear from you?
2. **AGENT** `2469338` 2017-11-15 10:08: @639919 It can take up to 28 days for a full response to be issued ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `service_status_delay_enquiry`, margin 0.1578)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 2469338, information_provided] "It can take up to 28 days for a full response to be issued"

Fill in CSV row `case_2469339`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 004 · `case_2384958`

**First customer message:** @VirginTrains How many coaches will there be on 9S70 EUS - EDB and 9S77 EUS - GLC today please?

**Conversation**

1. **CUSTOMER** `2384958` 2017-10-20 09:03: @VirginTrains How many coaches will there be on 9S70 EUS - EDB and 9S77 EUS - GLC today please?
2. **AGENT** `2384956` 2017-10-20 09:08: @243044 10 carriages on both ^LC
3. **CUSTOMER** `2384957` 2017-10-20 09:21: @VirginTrains Great, thanks 🚊

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `service_status_delay_enquiry`, margin 0.0045)  
> Auto resolution: `other`, resolved: True, DM redirect: False, outcome `customer_confirmed`  
> Type: other. Outcome: customer_confirmed. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 2384956, other] "10 carriages on both"

Fill in CSV row `case_2384958`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 005 · `case_1645235`

**First customer message:** @VirginTrains the team at Euston were pretty good on balance; however one guy was awful. Not to me, but to one of the other passengers.

**Conversation**

1. **CUSTOMER** `1645235` 2017-11-06 01:37: @VirginTrains the team at Euston were pretty good on balance; however one guy was awful. Not to me, but to one of the other passengers.
2. **AGENT** `1645234` 2017-11-06 01:43: @502743 Hi, Sauj, this is not what we like to hear. Please use this link: https://t.co/nYLtD42Xps to make a formal complaint. ^PH

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `journey_disruption_complaint`, margin 0.0104)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: self_service. Outcome: redirected_to_channel. Brand agent turns: 1. Signals seen: self_service. Evidence: [tweet 1645234, self_service] "Please use this link: https://t.co/nYLtD42Xps to make a formal complaint."

Fill in CSV row `case_1645235`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 006 · `case_2401054`

**First customer message:** @VirginTrains really not appreciating the service in the West Coast line. It’s always take, never give with you guys.

**Conversation**

1. **CUSTOMER** `2401054` 2017-10-20 12:44: @VirginTrains really not appreciating the service in the West Coast line. It’s always take, never give with you guys.
2. **AGENT** `2401052` 2017-10-20 12:46: @690830 What's happened? ^MW
3. **CUSTOMER** `2401053` 2017-10-20 12:49: @VirginTrains A series of things lately. As a regular user not great. Is flying really going to be a better option going forwards?
4. **CUSTOMER** `2401051` 2017-10-20 12:52: @VirginTrains Shall we take this to DM?
5. **AGENT** `2401050` 2017-10-20 12:53: @690830 Can do yes ^MW https://t.co/hhfk9c3ylv

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `chitchat_non_support`, margin 0.0686)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: self_service. Outcome: redirected_to_channel. Brand agent turns: 2. Signals seen: self_service, clarification_requested. Evidence: [tweet 2401050, self_service] "Can do yes ^MW https://t.co/hhfk9c3ylv"

Fill in CSV row `case_2401054`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 007 · `case_1611232`

**First customer message:** Been stuck at Watford Junction over an hour and half @LondonMidland, how much longer do I have to wait? @VirginTrains @115793

**Conversation**

1. **CUSTOMER** `1611232` 2017-11-05 11:45: Been stuck at Watford Junction over an hour and half @LondonMidland, how much longer do I have to wait? @VirginTrains @115793
2. **OTHER-AGENT** `1611233` 2017-11-05 11:46: @494533 @VirginTrains @115793 Where are you heading?
3. **CUSTOMER** `1611368` 2017-11-05 11:47: @LondonMidland @VirginTrains @115793 I'm heading back to Macclesfield
4. **AGENT** `1611231` 2017-11-05 11:47: @494533 @LondonMidland @115793 So sorry for your experience, Carissa, we hope to have you on the move shortly ^HP
5. **CUSTOMER** `1611230` 2017-11-05 11:49: @VirginTrains @LondonMidland @115793 How long is shortly? I need to get back to Macclesfield.
6. **AGENT** `1611228` 2017-11-05 11:57: @494533 @LondonMidland @115793 We are awaiting further updates but you can check our Live Updates here: https://t.co/NGZTHOZj8J
7. **CUSTOMER** `1611229` 2017-11-05 11:58: @VirginTrains @LondonMidland @115793 We've been redirected to a coach to Milton Keynes after a long two hour wait at the station

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 2 `delay_in_progress_report`; runner-up `service_status_delay_enquiry`, margin 0.0209)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: self_service. Outcome: customer_followup_unanswered. Brand agent turns: 2. Signals seen: self_service, information_provided. Evidence: [tweet 1611228, self_service] "We are awaiting further updates but you can check our Live Updates here: https://t.co/NGZTHOZj8J"

Fill in CSV row `case_1611232`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 008 · `case_539568`

**First customer message:** @VirginTrains I am due to catch the 16.43 from Euston to Birmingham New Street is that train running as I can’t see it listed

**Conversation**

1. **CUSTOMER** `539568` 2017-12-02 14:21: @VirginTrains I am due to catch the 16.43 from Euston to Birmingham New Street is that train running as I can’t see it listed
2. **AGENT** `539567` 2017-12-02 14:44: @245129 This service is expected to run ^MM
3. **CUSTOMER** `539566` 2017-12-02 14:47: @VirginTrains Thank you what does ^MM mean?
4. **AGENT** `539564` 2017-12-02 14:52: @245129 These are my initials, Louise ^MM
5. **CUSTOMER** `539565` 2017-12-02 16:35: @VirginTrains Oh never thought of that thought it was some code thank you 😊

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 1 `service_running_enquiry`; runner-up `service_status_delay_enquiry`, margin 0.174)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: information_provided. Outcome: customer_followup_unanswered. Brand agent turns: 2. Signals seen: information_provided, other. Evidence: [tweet 539567, information_provided] "This service is expected to run"

Fill in CSV row `case_539568`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 009 · `case_482906`

**First customer message:** @VirginTrains I’m meant to be travelling from Chester to London on the 15th December at half 4 and the website says you will not be serving Chester on this day. What is going on instead?

**Conversation**

1. **CUSTOMER** `482906` 2017-12-01 13:54: @VirginTrains I’m meant to be travelling from Chester to London on the 15th December at half 4 and the website says you will not be serving Chester on this day. What is going on instead?
2. **AGENT** `482904` 2017-12-01 14:08: @229855 Apologies Jamie, please keep an eye on our website as we will provide further updates on alternative travel arrangements ^HP
3. **CUSTOMER** `482905` 2017-12-01 14:21: @VirginTrains Thank you for the quick response. It’s ok. I’ve made alternative arrangements of my own. How would I go about claiming a refund please?

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 2 `delay_in_progress_report`; runner-up `service_status_delay_enquiry`, margin 0.0273)  
> Auto resolution: `self_service`, resolved: True, DM redirect: False, outcome `customer_confirmed`  
> Type: self_service. Outcome: customer_confirmed. Brand agent turns: 1. Signals seen: self_service. Evidence: [tweet 482904, self_service] "Apologies Jamie, please keep an eye on our website as we will provide further updates on alternative travel arrangements"

Fill in CSV row `case_482906`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 010 · `case_558860`

**First customer message:** @250609 @VirginTrains Hahahahaha

**Conversation**

1. **CUSTOMER** `558860` 2017-12-02 22:08: @250609 @VirginTrains Hahahahaha

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `unclear_or_media_only` (cluster -1 `unclear_or_media_only`; runner-up `unclear_or_media_only`, margin 0.0518)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `no_response`  
> Brand never replied in the public thread (outcome: no_response).

Fill in CSV row `case_558860`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 011 · `case_1303135`

**First customer message:** @VirginTrains time to think about #plasticwaste on trains. Most items in photo will be around for 100 years+. Time for tea in a mug? https://t.co/foqBkHVTm7

**Conversation**

1. **CUSTOMER** `1303135` 2017-10-16 09:50: @VirginTrains time to think about #plasticwaste on trains. Most items in photo will be around for 100 years+. Time for tea in a mug? https://t.co/foqBkHVTm7
2. **AGENT** `1303134` 2017-10-16 09:56: @424527 Good morning Phil. We will be sure to pass your feedback onto the relevant team. ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `chitchat_non_support`, margin 0.0104)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: feedback_acknowledged. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: feedback_acknowledged. Evidence: [tweet 1303134, feedback_acknowledged] "We will be sure to pass your feedback onto the relevant team."

Fill in CSV row `case_1303135`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 012 · `case_115458`

**First customer message:** @VirginTrains Hi! Is the Black Friday sale only for tickets between 18th Dec and 14th January? The East Coast sale was from 6th Jan to 2nd March and I assumed they would be the same.

**Conversation**

1. **CUSTOMER** `115458` 2017-11-23 21:21: @VirginTrains Hi! Is the Black Friday sale only for tickets between 18th Dec and 14th January? The East Coast sale was from 6th Jan to 2nd March and I assumed they would be the same.
2. **AGENT** `115456` 2017-11-23 21:22: @141662 This is correct Stuart, the 22/12, 23/12 and 27/12 aren't included in the sale either. As we run different lines we have different sale dates. ^BT
3. **CUSTOMER** `115457` 2017-11-23 21:24: @VirginTrains ok, thanks 🙂

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `service_status_delay_enquiry`, margin 0.1257)  
> Auto resolution: `information_provided`, resolved: True, DM redirect: False, outcome `customer_confirmed`  
> Type: information_provided. Outcome: customer_confirmed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 115456, information_provided] "This is correct Stuart, the 22/12, 23/12 and 27/12 aren't included in the sale either. As we run different lines we have different sale dates."

Fill in CSV row `case_115458`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 013 · `case_2614736`

**First customer message:** @VirginTrains #excellent service from Alan&amp;david #1stclassleedstolo don

**Conversation**

1. **CUSTOMER** `2614736` 2017-11-17 13:39: @VirginTrains #excellent service from Alan&amp;david #1stclassleedstolo don
2. **AGENT** `2614734` 2017-11-17 13:41: @739137 We'll pass on your kind words to the @120576 team ^CB
3. **OTHER-AGENT** `2614735` 2017-11-17 13:48: @VirginTrains @739137 Great to hear Keli - what time service are you on? ^KM

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `praise_positive_feedback`, margin 0.0506)  
> Auto resolution: `redirected_to_other_operator`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: redirected_to_other_operator. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: redirected_to_other_operator, feedback_acknowledged. Evidence: [tweet 2614734, redirected_to_other_operator] "We'll pass on your kind words to the @120576 team"

Fill in CSV row `case_2614736`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 014 · `case_993683`

**First customer message:** @VirginTrains Pre booked seats 1st class only to see they have double booked no staff to rectify or ask very disappointed.

**Conversation**

1. **CUSTOMER** `993683` 2017-10-22 15:52: @VirginTrains Pre booked seats 1st class only to see they have double booked no staff to rectify or ask very disappointed.
2. **AGENT** `993681` 2017-10-22 15:56: @355399 Oh no, which service are you on? ^PA
3. **CUSTOMER** `993682` 2017-10-22 16:03: @VirginTrains West coast - Euston to Stockport 16.57
4. **AGENT** `993684` 2017-10-22 16:06: @355399 There should be a member of staff in coach C who you can speak to. ^PA

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `first_class_catering_issue`, margin 0.0177)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: self_service. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: self_service, clarification_requested. Evidence: [tweet 993684, self_service] "There should be a member of staff in coach C who you can speak to."

Fill in CSV row `case_993683`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 015 · `case_70861`

**First customer message:** @VirginTrains On the 9:15 to Euston from Man Piccadilly and the train manager, Alex is awesome - funny and lively. Do pass on my comments to his manager. Thanks.

**Conversation**

1. **CUSTOMER** `70861` 2017-11-30 09:48: @VirginTrains On the 9:15 to Euston from Man Piccadilly and the train manager, Alex is awesome - funny and lively. Do pass on my comments to his manager. Thanks.
2. **AGENT** `70860` 2017-11-30 09:58: @131913 Thanks for this feedback Imran, we will be sure to pass your comments on regarding this. ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `praise_positive_feedback`, margin 0.0006)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: feedback_acknowledged. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: feedback_acknowledged. Evidence: [tweet 70860, feedback_acknowledged] "Thanks for this feedback Imran, we will be sure to pass your comments on regarding this."

Fill in CSV row `case_70861`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 016 · `case_2323653`

**First customer message:** @VirginTrains poor poor customer service from northampton - Birmingham new street #delayed

**Conversation**

1. **CUSTOMER** `2323653` 2017-12-02 12:32: @VirginTrains poor poor customer service from northampton - Birmingham new street #delayed
2. **AGENT** `2323652` 2017-12-02 12:58: @673242 Hi Tia, which service were you travelling on please? ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `service_status_delay_enquiry`, margin 0.0283)  
> Auto resolution: `clarification_requested`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: clarification_requested. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: clarification_requested. Evidence: [tweet 2323652, clarification_requested] "Hi Tia, which service were you travelling on please?"

Fill in CSV row `case_2323653`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 017 · `case_1238516`

**First customer message:** Chaotic handling of delays by @VirginTrains at Crewe people told to change platforms last minute. People running with their belongings.👎👎

**Conversation**

1. **CUSTOMER** `1238516` 2017-10-25 21:30: Chaotic handling of delays by @VirginTrains at Crewe people told to change platforms last minute. People running with their belongings.👎👎
2. **AGENT** `1238515` 2017-10-25 21:34: @410516 Sorry for your experience, Arshad. We'll pass your comments on regarding this situation ^MM

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 2 `delay_in_progress_report`; runner-up `service_status_delay_enquiry`, margin 0.0111)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: feedback_acknowledged. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: feedback_acknowledged. Evidence: [tweet 1238515, feedback_acknowledged] "We'll pass your comments on regarding this situation"

Fill in CSV row `case_1238516`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 018 · `case_570653`

**First customer message:** @VirginTrains my train from Manchester leaves at 5:35 tomorrow to go to London. What are the chances it's going to run because of the Wembley situation? I already had a cancelled train yesterday

**Conversation**

1. **CUSTOMER** `570653` 2017-12-03 03:57: @VirginTrains my train from Manchester leaves at 5:35 tomorrow to go to London. What are the chances it's going to run because of the Wembley situation? I already had a cancelled train yesterday
2. **AGENT** `570652` 2017-12-03 03:58: @253997 It should be fine ^OL

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 1 `service_running_enquiry`; runner-up `journey_disruption_complaint`, margin 0.1409)  
> Auto resolution: `other`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: other. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 570652, other] "It should be fine"

Fill in CSV row `case_570653`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 019 · `case_1236545`

**First customer message:** Our @VirginTrains is optimistically heading towards London, knowing that the Milton Keynes-London line is closed due to an incident...

**Conversation**

1. **CUSTOMER** `1236545` 2017-10-25 18:23: Our @VirginTrains is optimistically heading towards London, knowing that the Milton Keynes-London line is closed due to an incident...
2. **AGENT** `1236543` 2017-10-25 18:46: @123240 Some lines have now reopened to/from London Euston. Delays are expected until the end of service ^MM
3. **CUSTOMER** `1236544` 2017-10-25 19:08: @VirginTrains Seems open, but why are additional stops being added along the way? Now stopping at Watford junction, causing an increased delay.
4. **AGENT** `1236546` 2017-10-25 19:12: @123240 This is due to previous services being cancelled to ensure all passengers can get to their destinations ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 9 `service_status_update_request`; runner-up `praise_positive_feedback`, margin 0.0163)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: information_provided. Evidence: [tweet 1236543, information_provided] "Some lines have now reopened to/from London Euston. Delays are expected until the end of service"; [tweet 1236546, information_provided] "This is due to previous services being cancelled to ensure all passengers can get to their destinations"

Fill in CSV row `case_1236545`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 020 · `case_1898483`

**First customer message:** TMW you're happy about having booked a seat with a plug &amp; table .... and then this .... @VirginTrains 😩 https://t.co/lJz4U59msN

**Conversation**

1. **CUSTOMER** `1898483` 2017-10-18 12:52: TMW you're happy about having booked a seat with a plug &amp; table .... and then this .... @VirginTrains 😩 https://t.co/lJz4U59msN
2. **AGENT** `1898482` 2017-10-18 12:54: @565807 What issues are you having? ^PA
3. **CUSTOMER** `1898480` 2017-10-18 12:55: @VirginTrains there's no power.
4. **AGENT** `1898478` 2017-10-18 12:58: @565807 Apologies! If you have a word with the Train Manager onboard, they may be ablel to reboot the socket for you ^KS
5. **CUSTOMER** `1898479` 2017-10-18 13:00: @VirginTrains cheers will do if I see her. To be honest it means I have an excuse not to work 🤣🤣 so won't try too hard 😜
6. **AGENT** `1898481` 2017-10-18 13:02: @565807 Just sit back, relax and enjoy the ride 😉 ^KS

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `praise_positive_feedback`, margin 0.1731)  
> Auto resolution: `troubleshooting`, resolved: True, DM redirect: False, outcome `agent_closed`  
> Type: troubleshooting. Outcome: agent_closed. Brand agent turns: 3. Signals seen: troubleshooting, information_provided, clarification_requested. Evidence: [tweet 1898478, troubleshooting] "If you have a word with the Train Manager onboard, they may be ablel to reboot the socket for you"

Fill in CSV row `case_1898483`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 021 · `case_499710`

**First customer message:** @VirginTrains got first class ticket and my plug isn’t working. I asked to get a different seat. I was ignored. WiFi is running too slow. And this is the food I got. Not very first class is it? #virgintrains #scam https://t.co/qbOQHDLgje

**Conversation**

1. **CUSTOMER** `499710` 2017-12-01 18:56: @VirginTrains got first class ticket and my plug isn’t working. I asked to get a different seat. I was ignored. WiFi is running too slow. And this is the food I got. Not very first class is it? #virgintrains #scam https://t.co/qbOQHDLgje
2. **AGENT** `499709` 2017-12-01 19:18: @234349 If you mention this to the team onboard, they may be able to reset this for you ^MM

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `onboard_wifi_issue` (cluster 11 `onboard_wifi_issue`; runner-up `first_class_catering_issue`, margin 0.0399)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 499709, information_provided] "If you mention this to the team onboard, they may be able to reset this for you"

Fill in CSV row `case_499710`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 022 · `case_1340104`

**First customer message:** @VirginTrains I assume that’s a we don’t know!

**Conversation**

1. **CUSTOMER** `1340104` 2017-10-27 15:07: @VirginTrains I assume that’s a we don’t know!
2. **AGENT** `1340103` 2017-10-27 15:24: @432570 Apologies Peter, is there anything we can help with? ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `customer_service_complaint`, margin 0.0861)  
> Auto resolution: `other`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: other. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 1340103, other] "Apologies Peter, is there anything we can help with?"

Fill in CSV row `case_1340104`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 023 · `case_2280164`

**First customer message:** Hey @VirginTrains I booked train tickets a couple of hours ago for tomorrow and no email confirmation yet. Can you help?

**Conversation**

1. **CUSTOMER** `2280164` 2017-10-19 18:55: Hey @VirginTrains I booked train tickets a couple of hours ago for tomorrow and no email confirmation yet. Can you help?
2. **AGENT** `2280162` 2017-10-19 18:59: @662863 Hi there, can you please send us your email address you booked via and we'll take a look for you? Thanks ^HP
3. **CUSTOMER** `2280163` 2017-10-19 19:00: @VirginTrains __email__
4. **AGENT** `2280165` 2017-10-19 19:06: @662863 Apologies Charlotte, the only tickets I can see on your account are for journeys made on 12th October I'm afraid ^HP
5. **CUSTOMER** `2280166` 2017-10-19 19:11: @VirginTrains Well, my card has been charged and I have a photograph of the screen confirming collection details so they've definitely been booked!
6. **AGENT** `2280167` 2017-10-19 19:13: @662863 We'd advise contacting our Aftersales team on 0344 556 5650 and they should be able to assist further, Charlotte ^HP
7. **CUSTOMER** `2280168` 2017-10-19 19:18: @VirginTrains Do I need email confirmation to collect ticket or is the reference and my card details enough?
8. **AGENT** `2280170` 2017-10-19 19:21: @662863 1/2 You will only need the collection reference number and you card to collect the tickets, Charlotte, but we'd advise
9. **AGENT** `2280169` 2017-10-19 19:22: @662863 2/2 just double checking the booking with our Aftersales team in case there are any issues ^HP
10. **CUSTOMER** `2280171` 2017-10-19 19:22: @VirginTrains Ok. Thanks. Will call them before I set off to check

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `delay_repay_refund_claim`, margin 0.0384)  
> Auto resolution: `information_provided`, resolved: True, DM redirect: False, outcome `customer_confirmed`  
> Type: information_provided. Outcome: customer_confirmed. Brand agent turns: 5. Signals seen: information_provided, clarification_requested. Evidence: [tweet 2280165, information_provided] "Apologies Charlotte, the only tickets I can see on your account are for journeys made on 12th October I'm afraid"; [tweet 2280167, information_provided] "We'd advise contacting our Aftersales team on 0344 556 5650 and they should be able to assist further, Charlotte"; [tweet 2280170, information_provided] "You will only need the collection reference number and you card to collect the tickets, Charlotte, but we'd advise"; [tweet 2280169, information_provided] "just double checking the booking with our Aftersales team in case there are any issues"

Fill in CSV row `case_2280164`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 024 · `case_1351191`

**First customer message:** @120576 any chance of holding the 19.30 from Edinburgh to Glasgow due to the persistent delays today??

**Conversation**

1. **CUSTOMER** `1351191` 2017-10-27 17:46: @120576 any chance of holding the 19.30 from Edinburgh to Glasgow due to the persistent delays today??
2. **OTHER-AGENT** `1351189` 2017-10-27 17:52: @435082 Hi Lucy, that sounds like a @VirginTrains service. We'll make the team aware for you. ^JC
3. **AGENT** `1351188` 2017-10-27 17:54: @435082 Thanks @120576. We wouldn't be able to hold this service unfortunately. Sorry for this Lucy. ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 9 `service_status_update_request`; runner-up `service_status_delay_enquiry`, margin 0.1095)  
> Auto resolution: `redirected_to_other_operator`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: redirected_to_other_operator. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: redirected_to_other_operator. Evidence: [tweet 1351188, redirected_to_other_operator] "We wouldn't be able to hold this service unfortunately."

Fill in CSV row `case_1351191`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 025 · `case_635156`

**First customer message:** What is the point of having a seat reservation system if it never works? @VirginTrains

**Conversation**

1. **CUSTOMER** `635156` 2017-11-22 17:00: What is the point of having a seat reservation system if it never works? @VirginTrains
2. **AGENT** `635154` 2017-11-22 17:03: @270601 Which service are you on today exactly? ^MW
3. **CUSTOMER** `635155` 2017-11-22 18:15: @VirginTrains Haymarket too Birmingham new street....no hot drinks or food....and now stuck at Carlisle
4. **AGENT** `635157` 2017-11-22 18:16: @270601 Sorry to hear that, there is flooding at the moment, we have road transport available from Lancaster to Preston in place ^MW
5. **CUSTOMER** `635158` 2017-11-22 18:52: @VirginTrains Any update?
6. **AGENT** `635159` 2017-11-22 18:53: @270601 Still the same situation I'm afraid :-( ^MW
7. **CUSTOMER** `635160` 2017-11-22 21:03: @VirginTrains Are trains still running from Preston the Birmingham new street?
8. **AGENT** `635161` 2017-11-22 21:08: @270601 Yes there will be a 21:17 service ^MW
9. **CUSTOMER** `635162` 2017-11-22 21:09: @VirginTrains I hope it's going to wait for the coaches to arrive before leaving.
10. **AGENT** `635163` 2017-11-22 21:18: @270601 There are extra coaches, taxis and mini buses ordered for passengers travelling between Lancaster and Preston. ^BT
11. **CUSTOMER** `635164` 2017-11-22 21:19: @VirginTrains That's not what I asked. Im on a coach from Lancaster to Preston....need to get to Birmingham.....is the 21.17 waiting for the coaches?
12. **AGENT** `635165` 2017-11-22 21:21: @270601 Afraid not, please speak to the team at Preston who will get you moving by road if no trains available ^MW
13. **CUSTOMER** `635166` 2017-11-22 23:19: @VirginTrains So now you have around 100 customers at Crewe all waiting for some service to Birmingham?? What's going on?? Staff here don't know??
14. **AGENT** `635167` 2017-11-22 23:31: @270601 There is a train on platform 5 due to go forward to Birmingham New Street now. ^BT
15. **CUSTOMER** `635168` 2017-11-22 23:31: @VirginTrains Staff here are telling everyone only going as far as Wolverhampton?
16. **AGENT** `635169` 2017-11-22 23:34: @270601 Yes, sorry we have just been told this will go forward to Wolverhampton. And onward travel from Wolverhampton has been arranged to take passengers forward to BHM. ^BT
17. **CUSTOMER** `635170` 2017-11-23 00:36: @VirginTrains Same but now at Wolverhampton 😴 https://t.co/PHSLRHPuQA
18. **CUSTOMER** `635171` 2017-11-23 16:27: @VirginTrains Guess what......reserved seat for 4.30 from Birmingham........and the coach isn't even here? 😂🤦🏼‍♂️
19. **AGENT** `635172` 2017-11-23 16:31: @270601 Oh no, do you have seats onboard at all? ^MW
20. **CUSTOMER** `635173` 2017-11-23 16:33: @VirginTrains There are 5 carriages rather than 10.....so no chance. Worryingly the driver didn't know there was only going to be 5 carriages?
21. **AGENT** `635174` 2017-11-23 16:49: @270601 I see, sorry to hear that, if you didn't get a seat please make a claim here - https://t.co/t20UbyOCSn ^MW

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `ticket_booking_query`, margin 0.2188)  
> Auto resolution: `compensation`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: compensation. Outcome: redirected_to_channel. Brand agent turns: 10. Signals seen: compensation, self_service, information_provided, clarification_requested. Evidence: [tweet 635174, compensation] "I see, sorry to hear that, if you didn't get a seat please make a claim here - https://t.co/t20UbyOCSn"

Fill in CSV row `case_635156`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 026 · `case_756981`

**First customer message:** @VirginTrains @300908 @120576 Sounds like a change in Man Picc. northern/eastmidlands combo.

**Conversation**

1. **CUSTOMER** `756981` 2017-10-11 18:56: @VirginTrains @300908 @120576 Sounds like a change in Man Picc. northern/eastmidlands combo.

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 9 `service_status_update_request`; runner-up `praise_positive_feedback`, margin 0.02)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `no_response`  
> Brand never replied in the public thread (outcome: no_response).

Fill in CSV row `case_756981`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 027 · `case_1100239`

**First customer message:** @VirginTrains Is there no Virgin service from Birmingham to London this morning?

**Conversation**

1. **CUSTOMER** `1100239` 2017-10-15 06:46: @VirginTrains Is there no Virgin service from Birmingham to London this morning?
2. **AGENT** `1100237` 2017-10-15 06:49: @379595 First service is at 0830 ^LC
3. **CUSTOMER** `1100238` 2017-10-15 06:49: @VirginTrains Ah. It's not showing up on trainline!

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 2 `delay_in_progress_report`; runner-up `customer_service_complaint`, margin 0.0077)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: unresolved. Outcome: customer_followup_unanswered. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 1100237, other] "First service is at 0830"

Fill in CSV row `case_1100239`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 028 · `case_2538820`

**First customer message:** Shit start at the train station. I made it in time for the train but just not enough time to “print” the &lt;paper ticket&gt;! Had the booking on my phone but they wouldn’t let me on. Thanks a lot @VirginTrains Isn’t it time to use upto date technology? It’s not like I hadn’t paid 🤬

**Conversation**

1. **CUSTOMER** `2538820` 2017-11-16 09:24: Shit start at the train station. I made it in time for the train but just not enough time to “print” the &lt;paper ticket&gt;! Had the booking on my phone but they wouldn’t let me on. Thanks a lot @VirginTrains Isn’t it time to use upto date technology? It’s not like I hadn’t paid 🤬
2. **AGENT** `2538819` 2017-11-16 09:29: @722064 1/2 Sorry to hear this, Zahira, I'm afraid we're not able to accept booking confirmation in place of actual tickets, however,
3. **AGENT** `2538821` 2017-11-16 09:29: @722064 2/2 you can select mobile tickets for future bookings if this is easier :) ^HP
4. **CUSTOMER** `2538818` 2017-11-16 09:33: @VirginTrains I would of expected some lenience in these sort of circumstances. When I booked the ticket the phone option wasn’t available.
5. **AGENT** `2538817` 2017-11-16 09:36: @722064 Sorry to hear that, Zahirah, if you book via our website or app, you can select M-Tickets or E-tickets ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `service_status_delay_enquiry`, margin 0.032)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: self_service. Outcome: agent_answered_unconfirmed. Brand agent turns: 3. Signals seen: self_service, information_provided. Evidence: [tweet 2538817, self_service] "Sorry to hear that, Zahirah, if you book via our website or app, you can select M-Tickets or E-tickets"

Fill in CSV row `case_2538820`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 029 · `case_1227507`

**First customer message:** @VirginTrains Train running 85 mins late...but staff are too lazy to serve customers. 12 staff partying.

**Conversation**

1. **CUSTOMER** `1227507` 2017-10-25 19:45: @VirginTrains Train running 85 mins late...but staff are too lazy to serve customers. 12 staff partying.
2. **AGENT** `1227505` 2017-10-25 19:55: @407536 Hi there, can you explain can you explain further on what you mean by this please? ^HP
3. **CUSTOMER** `1227506` 2017-10-25 19:56: @VirginTrains My husband is trying to get home .the staff are refusing to serve customers ?train delayed. He's been out since 6am?
4. **CUSTOMER** `1227508` 2017-10-25 19:57: @VirginTrains 12 staff sat in first....
5. **AGENT** `1228279` 2017-10-25 20:05: @407536 We'd advise your husband speaks to the staff onboard, Cecy. We'll however pass your comments on regarding this situation ^MM

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `first_class_catering_issue`, margin 0.0519)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: feedback_acknowledged. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: feedback_acknowledged, clarification_requested. Evidence: [tweet 1228279, feedback_acknowledged] "We'll however pass your comments on regarding this situation"

Fill in CSV row `case_1227507`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 030 · `case_2899906`

**First customer message:** @VirginTrains went up to the desk to ask the time of the next train &amp; got put through a third degree I’m VERY unimpressed with your staff &amp; how they accused me of being a liar!

**Conversation**

1. **CUSTOMER** `2899906` 2017-11-28 17:47: @VirginTrains went up to the desk to ask the time of the next train &amp; got put through a third degree I’m VERY unimpressed with your staff &amp; how they accused me of being a liar!
2. **AGENT** `2899904` 2017-11-28 17:54: @803350 Not great to hear, Xaviera, which station was this at please? And can you explain further on what happened? ^HP
3. **CUSTOMER** `2899905` 2017-11-28 17:57: @VirginTrains Wolverhampton. The woman basically said ‘well my friend said you’ve just been up to buy a ticket for Dudley port’ when I hadn’t been up nor am I going to Dudley port &amp; refused to give me the train times too. She was very rude in how she accused me of being up already
4. **CUSTOMER** `2899907` 2017-11-28 17:57: @VirginTrains When I hadn’t even been up! I couldn’t even see the other cashier! Clearly they got it wrong but they shouldn’t be so adamant
5. **AGENT** `2899908` 2017-11-28 18:00: @803350 Sorry to hear about your experience, have you been able to get onboard now at all? ^HP
6. **CUSTOMER** `2899909` 2017-11-28 18:02: @VirginTrains Yes but with a lot of stress. I suffer with anxiety and this hasn’t helped at all and it’s put me off using Virgin now after this experience

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `customer_service_complaint`, margin 0.1097)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: unresolved. Outcome: customer_followup_unanswered. Brand agent turns: 2. Signals seen: clarification_requested. Evidence: [tweet 2899908, clarification_requested] "Sorry to hear about your experience, have you been able to get onboard now at all?"

Fill in CSV row `case_2899906`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 031 · `case_1723492`

**First customer message:** @VirginTrains I booked tickets on your website for Edinburgh to London on Friday 3rd and train was cancelled where do I claim a refund?

**Conversation**

1. **CUSTOMER** `1723492` 2017-11-07 09:12: @VirginTrains I booked tickets on your website for Edinburgh to London on Friday 3rd and train was cancelled where do I claim a refund?
2. **AGENT** `1723490` 2017-11-07 09:14: @521363 Hi Mark, what tickets did you have and did you abandon travel all together? ^BT
3. **CUSTOMER** `1723491` 2017-11-07 10:05: @VirginTrains I had first class tickets on the 1pm train and we had to get a later train that was also late so messed up our travel plans
4. **AGENT** `1723493` 2017-11-07 10:08: @521363 I see, thank you. If you follow this link, https://t.co/QADF8nTSSt you can claim there. ^BT
5. **CUSTOMER** `1723494` 2017-11-07 10:15: @VirginTrains That just takes me to a page about how to save on tickets, I can't see anywhere to make a claim
6. **AGENT** `1723495` 2017-11-07 10:20: @521363 Sorry Mark, not sure what happened there, this is the correct link https://t.co/bgLLjZKBkE ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `ticket_booking_query`, margin 0.131)  
> Auto resolution: `compensation`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: compensation. Outcome: redirected_to_channel. Brand agent turns: 3. Signals seen: compensation, self_service, clarification_requested. Evidence: [tweet 1723493, compensation] "If you follow this link, https://t.co/QADF8nTSSt you can claim there."

Fill in CSV row `case_1723492`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 032 · `case_1614342`

**First customer message:** Thanks to @VirginTrains and @123241 I’m currently enjoying the sites of the A1 #RoadTrip #YouOweMeMoney #Refund #OneJob https://t.co/0ZR5HNPaBK

**Conversation**

1. **CUSTOMER** `1614342` 2017-11-05 13:05: Thanks to @VirginTrains and @123241 I’m currently enjoying the sites of the A1 #RoadTrip #YouOweMeMoney #Refund #OneJob https://t.co/0ZR5HNPaBK
2. **AGENT** `1614343` 2017-11-05 13:13: @495182 @123241 1/2 Apologies Adam, we've unfortunately been affected by signalling failure on the lines today. You can make a claim
3. **AGENT** `1614341` 2017-11-05 13:13: @495182 @123241 2/2 for compensation here: https://t.co/MY9v6GHRxH ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `praise_positive_feedback`, margin 0.0138)  
> Auto resolution: `compensation`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: compensation. Outcome: redirected_to_channel. Brand agent turns: 2. Signals seen: compensation, self_service. Evidence: [tweet 1614343, compensation] "You can make a claim"; [tweet 1614341, compensation] "for compensation here: https://t.co/MY9v6GHRxH"

Fill in CSV row `case_1614342`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 033 · `case_705044`

**First customer message:** @VirginTrains can you change were you selected to pick your tickets up from?

**Conversation**

1. **CUSTOMER** `705044` 2017-11-23 18:53: @VirginTrains can you change were you selected to pick your tickets up from?
2. **AGENT** `705043` 2017-11-23 18:55: @288524 You can collect from any machine at any station, Nathanael ^MW

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `seat_reservation_issue`, margin 0.1826)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 705043, information_provided] "You can collect from any machine at any station, Nathanael"

Fill in CSV row `case_705044`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 034 · `case_2273124`

**First customer message:** @VirginTrains why can't I use my rail card when buying a ticket on the train?

**Conversation**

1. **CUSTOMER** `2273124` 2017-10-19 18:36: @VirginTrains why can't I use my rail card when buying a ticket on the train?
2. **AGENT** `2273123` 2017-10-19 18:37: @661241 Hi there, what type of railcard do you have please? And which service are you travelling on? ^HP
3. **CUSTOMER** `2273122` 2017-10-19 18:38: @VirginTrains I have a 16-25 railcars travelling from Edinburgh to Kings Cross 17.31
4. **AGENT** `2273121` 2017-10-19 18:41: @661241 Thanks for confirming, it sounds like you're travelling with @120576 so we'll pass this over to the team for you ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `ticket_booking_query`, margin 0.0206)  
> Auto resolution: `redirected_to_other_operator`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: redirected_to_other_operator. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: redirected_to_other_operator, clarification_requested. Evidence: [tweet 2273121, redirected_to_other_operator] "Thanks for confirming, it sounds like you're travelling with @120576 so we'll pass this over to the team for you"

Fill in CSV row `case_2273124`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 035 · `case_1156100`

**First customer message:** Really!? @VirginTrains what on earth makes you think mayonnaise is needed in a new pork wrap? Again no mayo free option ☹ you are obsessed!

**Conversation**

1. **CUSTOMER** `1156100` 2017-10-15 18:01: Really!? @VirginTrains what on earth makes you think mayonnaise is needed in a new pork wrap? Again no mayo free option ☹ you are obsessed!
2. **AGENT** `1156099` 2017-10-15 18:04: @391960 Pulled pork wrap, Steve? ^MW
3. **CUSTOMER** `1156098` 2017-10-15 18:06: @VirginTrains Yes, that is the one. Ta
4. **AGENT** `1156095` 2017-10-15 18:07: @391960 We'll pass this on for you, Steve ^MW
5. **CUSTOMER** `1156096` 2017-10-15 18:08: @VirginTrains 😁
6. **CUSTOMER** `1156097` 2017-10-15 19:08: @VirginTrains Your lovely staff made up for it whenand gave me whatever else they could get their hands on. Only so much pretzel and bisc can be eaten 😁

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `customer_service_complaint`, margin 0.0743)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: feedback_acknowledged. Outcome: customer_followup_unanswered. Brand agent turns: 2. Signals seen: feedback_acknowledged, other. Evidence: [tweet 1156095, feedback_acknowledged] "We'll pass this on for you, Steve"

Fill in CSV row `case_1156100`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 036 · `case_1968442`

**First customer message:** @VirginTrains hi BH

**Conversation**

1. **CUSTOMER** `1968442` 2017-11-07 22:17: @VirginTrains hi BH
2. **AGENT** `1968440` 2017-11-07 22:21: @583236 Hi dude ^BH
3. **CUSTOMER** `1968441` 2017-11-07 22:22: @VirginTrains Northern unit numbers 1055 Middlesbrough to Darlington its like 142018 as example BH
4. **AGENT** `1968443` 2017-11-07 22:24: @583236 SOrry? ^BH
5. **CUSTOMER** `1968445` 2017-11-07 22:25: @VirginTrains Unit number its like 142087 as example
6. **CUSTOMER** `1968444` 2017-11-07 22:26: @VirginTrains Have a look on internal system please
7. **AGENT** `1968446` 2017-11-07 22:28: @583236 What are you on about? ^BH
8. **CUSTOMER** `1968447` 2017-11-07 22:29: @VirginTrains U know on internal system unit number loco unit number say example unit number for 1055 Middlesbrough is 142079
9. **CUSTOMER** `1968448` 2017-11-07 22:36: @VirginTrains Have u found out
10. **AGENT** `1968449` 2017-11-07 22:40: @583236 Huh? ^BH
11. **CUSTOMER** `1968450` 2017-11-07 22:41: @VirginTrains Unit number like on front of train check where's 142020 at on internal system please and provide it
12. **CUSTOMER** `1968451` 2017-11-07 22:44: @VirginTrains Do u understand

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `unclear_or_media_only` (cluster -1 `unclear_or_media_only`; runner-up `unclear_or_media_only`, margin 0.0518)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: unresolved. Outcome: customer_followup_unanswered. Brand agent turns: 4. Signals seen: clarification_requested, other. Evidence: [tweet 1968449, other] "Huh?"

Fill in CSV row `case_1968442`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 037 · `case_1320758`

**First customer message:** From cancelled trains to spilled coffee, I’m on my way 2 Manchester via Crewe, thanks @VirginTrains &amp; @LondonMidland 🙋for getting me home

**Conversation**

1. **CUSTOMER** `1320758` 2017-10-27 08:58: From cancelled trains to spilled coffee, I’m on my way 2 Manchester via Crewe, thanks @VirginTrains &amp; @LondonMidland 🙋for getting me home
2. **AGENT** `1320756` 2017-10-27 09:01: @428382 @LondonMidland Sounds like a hectic morning. Glad to hear you're on the move ^CB
3. **CUSTOMER** `1320757` 2017-10-27 09:02: @VirginTrains @LondonMidland It’s been crazy, but I’m making my way! Happy Friday 🕺🕺🕺
4. **OTHER-AGENT** `1320759` 2017-10-27 09:05: @428382 @VirginTrains glad to hear you are on your way Nick
5. **CUSTOMER** `1320760` 2017-10-27 09:06: @LondonMidland @VirginTrains Thanks guys 🕺🕺

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `praise_positive_feedback` (cluster 8 `praise_positive_feedback`; runner-up `service_status_delay_enquiry`, margin 0.0229)  
> Auto resolution: `information_provided`, resolved: True, DM redirect: False, outcome `customer_confirmed`  
> Type: information_provided. Outcome: customer_confirmed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 1320756, information_provided] "Sounds like a hectic morning. Glad to hear you're on the move"

Fill in CSV row `case_1320758`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 038 · `case_2433758`

**First customer message:** @VirginTrains curious as to why coach G on 11.33 lpool to Euston train is not getting the complimentary scran that other coaches are getting

**Conversation**

1. **CUSTOMER** `2433758` 2017-10-29 13:29: @VirginTrains curious as to why coach G on 11.33 lpool to Euston train is not getting the complimentary scran that other coaches are getting
2. **AGENT** `2433757` 2017-10-29 13:35: @698403 Hi Nick. Have you spoke to a member of onboard staff regarding this? ^BT
3. **CUSTOMER** `2433759` 2017-10-29 13:39: @VirginTrains Nope, train manager is nowhere to be found
4. **AGENT** `2434482` 2017-10-29 13:46: @698403 They would be going through each coach throughout the journey ^CB
5. **CUSTOMER** `2434481` 2017-10-29 13:49: @VirginTrains Now spoken to a member of staff &amp;they were told specifically not to serve our coach. What’s the point in paying extra for 1st class?
6. **AGENT** `2434480` 2017-10-29 13:52: @698403 The service may have been declassified in Coach G. Did the team advise of this? ^CB
7. **CUSTOMER** `2434478` 2017-10-29 13:56: @VirginTrains Nope, paid the £20 weekend upgrade sat in Coach G so definitely not declassified
8. **AGENT** `2434476` 2017-10-29 13:58: @698403 If you move in to another First Class coach they'd be able to assist further ^CB
9. **CUSTOMER** `2434477` 2017-10-29 14:02: @VirginTrains Weren’t enough seats in other carriages to sit with friends. Can I get my upgrade fee refunded?
10. **AGENT** `2434479` 2017-10-29 14:04: @698403 You'll need to contact our team through our website: https://t.co/CLAVO9iu36 ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `service_status_delay_enquiry`, margin 0.0132)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: self_service. Outcome: redirected_to_channel. Brand agent turns: 5. Signals seen: self_service, information_provided, clarification_requested, other. Evidence: [tweet 2434479, self_service] "You'll need to contact our team through our website: https://t.co/CLAVO9iu36"

Fill in CSV row `case_2433758`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 039 · `case_83997`

**First customer message:** @VirginTrains snacks galore! 🏴󠁧󠁢󠁳󠁣󠁴󠁿 https://t.co/qn92F6FhtC

**Conversation**

1. **CUSTOMER** `83997` 2017-11-30 13:36: @VirginTrains snacks galore! 🏴󠁧󠁢󠁳󠁣󠁴󠁿 https://t.co/qn92F6FhtC
2. **AGENT** `83996` 2017-11-30 13:53: @134457 Enjoy! ^LC

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `unclear_or_media_only` (cluster -1 `unclear_or_media_only`; runner-up `unclear_or_media_only`, margin 0.0518)  
> Auto resolution: `other`, resolved: True, DM redirect: False, outcome `agent_closed`  
> Type: other. Outcome: agent_closed. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 83996, other] "Enjoy!"

Fill in CSV row `case_83997`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 040 · `case_1389196`

**First customer message:** @VirginTrains what’s the process for refunds. Paid for 1st class on one train but forced to stand on another. Disappointing

**Conversation**

1. **CUSTOMER** `1389196` 2017-10-28 10:44: @VirginTrains what’s the process for refunds. Paid for 1st class on one train but forced to stand on another. Disappointing
2. **AGENT** `1389194` 2017-10-28 10:46: @443174 If you have a First Class ticket but don't receive the service please get in touch here: https://t.co/Ky8wEuC7hd ^CB
3. **CUSTOMER** `1389195` 2017-10-28 11:01: @VirginTrains Just tried using that form. Crashed twice then told me captcha isn’t working.....
4. **AGENT** `1389197` 2017-10-28 11:06: @443174 Can you try another device Usman? ^BT
5. **CUSTOMER** `1389198` 2017-10-28 11:09: @VirginTrains Could do....if I had one..... and to think that that virgin thinks it can run healthcare.......
6. **AGENT** `1389199` 2017-10-28 11:14: @443174 If you search for the Virgin Trains website, the same form will be on there. Maybe this will work on your device? ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `journey_disruption_complaint`, margin 0.0789)  
> Auto resolution: `troubleshooting`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: troubleshooting. Outcome: agent_awaiting_customer. Brand agent turns: 3. Signals seen: troubleshooting, self_service, clarification_requested, other. Evidence: [tweet 1389197, troubleshooting] "Can you try another device Usman?"

Fill in CSV row `case_1389196`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 041 · `case_1205574`

**First customer message:** @VirginTrains VT-021117-0505 after Euston being closed 27.10.17 so train- unable to get on any train north!4 hour delayed- when refund request I am told there was only a 21min delay so no refund?! #Lies

**Conversation**

1. **CUSTOMER** `1205574` 2017-11-10 13:18: @VirginTrains VT-021117-0505 after Euston being closed 27.10.17 so train- unable to get on any train north!4 hour delayed- when refund request I am told there was only a 21min delay so no refund?! #Lies
2. **AGENT** `1205573` 2017-11-10 13:20: @403159 Drop us a DM with this please, Mel ^MW

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `service_status_delay_enquiry`, margin 0.1301)  
> Auto resolution: `redirected_to_dm`, resolved: False, DM redirect: True, outcome `redirected_to_dm`  
> Type: redirected_to_dm. Outcome: redirected_to_dm. Brand agent turns: 1. Signals seen: redirected_to_dm. Evidence: [tweet 1205573, redirected_to_dm] "Drop us a DM with this please, Mel"

Fill in CSV row `case_1205574`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 042 · `case_1336623`

**First customer message:** Happy to sit on floor, but there's no room. And if u want to check my word for it on CCTV @VirginTrains - I'm in the carriage at the front!

**Conversation**

1. **CUSTOMER** `1336623` 2017-10-27 14:09: Happy to sit on floor, but there's no room. And if u want to check my word for it on CCTV @VirginTrains - I'm in the carriage at the front!
2. **AGENT** `1336633` 2017-10-27 14:20: @431569 1/2 Sorry about your experience today Emily. Many services are experiencing high volume of passengers due to disruption
3. **AGENT** `1336621` 2017-10-27 14:20: @431569 2/2 What service are you travelling on? ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `praise_positive_feedback`, margin 0.0675)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: information_provided. Outcome: agent_awaiting_customer. Brand agent turns: 2. Signals seen: information_provided, clarification_requested. Evidence: [tweet 1336633, information_provided] "Sorry about your experience today Emily. Many services are experiencing high volume of passengers due to disruption"

Fill in CSV row `case_1336623`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 043 · `case_1563013`

**First customer message:** Amazing, @VirginTrains won't help a girl out&amp;give me a spoon to eat my pret porridge!My mate did me a solid buying it but forgot the spoon

**Conversation**

1. **CUSTOMER** `1563013` 2017-11-04 10:20: Amazing, @VirginTrains won't help a girl out&amp;give me a spoon to eat my pret porridge!My mate did me a solid buying it but forgot the spoon
2. **AGENT** `1563010` 2017-11-04 10:26: @482704 Oh no, have you spoken to the onboard team, Laura? ^HP
3. **CUSTOMER** `1563012` 2017-11-04 10:28: @VirginTrains Yup, trolley guy I bought my drink from said he only had 1 so could I go to the train shop, train shop said no, you were losing too money
4. **CUSTOMER** `1563011` 2017-11-04 10:30: @VirginTrains Thinking of attempting to eat it with the lid of the small pot... what do you reckon? https://t.co/jB71FBvk7t
5. **AGENT** `1563015` 2017-11-04 10:32: @482704 Really sorry to hear that, Laura, which service are you travelling on? ^HP
6. **AGENT** `1563014` 2017-11-04 10:35: @482704 That might work though, Laura 🙈 ^HP
7. **CUSTOMER** `1563016` 2017-11-04 10:36: @VirginTrains 10:00am from London Euston to Manchester
8. **AGENT** `1563017` 2017-11-04 10:39: @482704 Did they have any of the plastic spoons left in the shop at all? ^HP
9. **CUSTOMER** `1563019` 2017-11-04 10:43: @VirginTrains I believe so, she didn't say, just no, they only have enough for the porridge is on board
10. **CUSTOMER** `1563018` 2017-11-04 10:43: @VirginTrains I've gone in, wish me luck #desperatetimes https://t.co/FxOvkRlbva
11. **AGENT** `1563020` 2017-11-04 10:46: @482704 Laura, you're my hero 🙏🏻 ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `first_class_catering_issue`, margin 0.1008)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 5. Signals seen: information_provided, clarification_requested, other. Evidence: [tweet 1563014, information_provided] "That might work though, Laura 🙈"

Fill in CSV row `case_1563013`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 044 · `case_480156`

**First customer message:** Why is it that every time I get on an @VirginTrains the seat reservation system is faulty. Russian roulette whether the seat I'm in is reserved

**Conversation**

1. **CUSTOMER** `480156` 2017-12-01 13:15: Why is it that every time I get on an @VirginTrains the seat reservation system is faulty. Russian roulette whether the seat I'm in is reserved
2. **AGENT** `480155` 2017-12-01 13:26: @229114 Not great to hear, Paul, apologies for the inconvenience caused. Have you been able to get your seat now? ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `ticket_booking_query`, margin 0.2187)  
> Auto resolution: `clarification_requested`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: clarification_requested. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: clarification_requested. Evidence: [tweet 480155, clarification_requested] "Have you been able to get your seat now?"

Fill in CSV row `case_480156`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 045 · `case_444036`

**First customer message:** Without fail everytime I try to book a bike on @VirginTrains there's a problem, mutiple lengthy phone calls or tickets can't be collected

**Conversation**

1. **CUSTOMER** `444036` 2017-11-22 09:14: Without fail everytime I try to book a bike on @VirginTrains there's a problem, mutiple lengthy phone calls or tickets can't be collected
2. **AGENT** `444035` 2017-11-22 09:16: @220194 What issues have you had Emily? ^PA
3. **CUSTOMER** `444034` 2017-11-22 09:20: @VirginTrains trying to book my bike on a train, the system is down, have to ring back in two hours, it is never possible to book bike with one phone call
4. **AGENT** `444033` 2017-11-22 09:23: @220194 If you DM us the details of the train we can have a look for bike spaces. ^PA

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `ticket_booking_query`, margin 0.0035)  
> Auto resolution: `redirected_to_dm`, resolved: False, DM redirect: True, outcome `redirected_to_dm`  
> Type: redirected_to_dm. Outcome: redirected_to_dm. Brand agent turns: 2. Signals seen: redirected_to_dm, clarification_requested. Evidence: [tweet 444033, redirected_to_dm] "If you DM us the details of the train we can have a look for bike spaces."

Fill in CSV row `case_444036`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 046 · `case_1265405`

**First customer message:** This is general advice but right now, it’s for you, @VirginTrains. https://t.co/iI1ikXmPz7

**Conversation**

1. **CUSTOMER** `1265405` 2017-10-26 12:20: This is general advice but right now, it’s for you, @VirginTrains. https://t.co/iI1ikXmPz7
2. **AGENT** `1265403` 2017-10-26 12:22: @416310 Are you struggling to connect or with speed? ^MW
3. **CUSTOMER** `1265404` 2017-10-26 12:27: @VirginTrains It supposedly connects immediately but every app just shows this or similar. Was working when I left Manc but after that, not so much 😕 https://t.co/ML9Led1VQw
4. **AGENT** `1265406` 2017-10-26 12:31: @416310 It might be worth speaking to Wi-Fi support on 0330 088 1271, Anthony ^MW

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `customer_service_complaint`, margin 0.0465)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: information_provided, clarification_requested. Evidence: [tweet 1265406, information_provided] "It might be worth speaking to Wi-Fi support on 0330 088 1271, Anthony"

Fill in CSV row `case_1265405`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 047 · `case_1408705`

**First customer message:** Thank you @VirginTrains for allowing me on an earlier train home as good will gesture.Great customer service at #kingscross from #virgintrains staff

**Conversation**

1. **CUSTOMER** `1408705` 2017-11-01 22:35: Thank you @VirginTrains for allowing me on an earlier train home as good will gesture.Great customer service at #kingscross from #virgintrains staff
2. **AGENT** `1408704` 2017-11-01 22:44: @122408 Great to hear, Allison! We've copied in @120576 so that they're aware of this 😀 ^PH

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `praise_positive_feedback`, margin 0.0019)  
> Auto resolution: `information_provided`, resolved: True, DM redirect: False, outcome `agent_closed`  
> Type: information_provided. Outcome: agent_closed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 1408704, information_provided] "Great to hear, Allison! We've copied in @120576 so that they're aware of this 😀"

Fill in CSV row `case_1408705`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 048 · `case_2125889`

**First customer message:** @VirginTrains just spent 32 mins on a call to your helpdesk only for them to hang up as soon as I got through #soannoyed

**Conversation**

1. **CUSTOMER** `2125889` 2017-11-08 17:49: @VirginTrains just spent 32 mins on a call to your helpdesk only for them to hang up as soon as I got through #soannoyed
2. **AGENT** `2125888` 2017-11-08 17:51: @273976 Sorry to hear this, which number had you contacted please? ^HP
3. **CUSTOMER** `2125887` 2017-11-08 18:00: @VirginTrains 03445565650
4. **AGENT** `2125886` 2017-11-08 18:05: @273976 We'll pass your comments on regarding this. Is there anything we can help you with? ^MM
5. **CUSTOMER** `2125884` 2017-11-08 18:14: @VirginTrains Basically my account booked 2 return tickets for the 15th of November even though the outwards journeys were on the 7th &amp; 9th. I've gone through the change process online but frustrating as I'm now £20 down.
6. **AGENT** `2125882` 2017-11-08 18:18: @273976 This can be escalated further with Customer Relations via the online complaints link - https://t.co/t20UbyOCSn ^MM
7. **CUSTOMER** `2125883` 2017-11-08 19:04: @VirginTrains Thanks I've sent the online form
8. **AGENT** `2125885` 2017-11-08 19:15: @273976 Thank you ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `service_status_delay_enquiry`, margin 0.0315)  
> Auto resolution: `escalated`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: escalated. Outcome: agent_answered_unconfirmed. Brand agent turns: 4. Signals seen: escalated, self_service, feedback_acknowledged, clarification_requested, other. Evidence: [tweet 2125882, escalated] "This can be escalated further with Customer Relations via the online complaints link - https://t.co/t20UbyOCSn"

Fill in CSV row `case_2125889`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 049 · `case_1333135`

**First customer message:** @VirginTrains you tell us to get onto Leeds train for M’ster then no staff and people demanding seats that should be declassified?

**Conversation**

1. **CUSTOMER** `1333135` 2017-10-27 12:59: @VirginTrains you tell us to get onto Leeds train for M’ster then no staff and people demanding seats that should be declassified?
2. **AGENT** `1333133` 2017-10-27 13:17: @431004 The @120576 would be best placed to help with the service to Leeds ^CB
3. **CUSTOMER** `1333134` 2017-10-27 13:31: @VirginTrains @120576 Totally unsuitable answer!
4. **OTHER-AGENT** `1333136` 2017-10-27 13:34: @431004 @VirginTrains Hi Pete, which service are you currently travelling? ^JC

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `seat_reservation_issue`, margin 0.0622)  
> Auto resolution: `redirected_to_other_operator`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: redirected_to_other_operator. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: redirected_to_other_operator. Evidence: [tweet 1333133, redirected_to_other_operator] "The @120576 would be best placed to help with the service to Leeds"

Fill in CSV row `case_1333135`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 050 · `case_1558990`

**First customer message:** @VirginTrains delayrepay remains unpaid after 25 days. Two emails to customer relations this week. No response at all. Dreadful service.

**Conversation**

1. **CUSTOMER** `1558990` 2017-11-04 13:06: @VirginTrains delayrepay remains unpaid after 25 days. Two emails to customer relations this week. No response at all. Dreadful service.
2. **AGENT** `1558988` 2017-11-04 13:08: @481789 Sorry to hear this, do you have a VT case reference number please? ^HP
3. **CUSTOMER** `1558989` 2017-11-04 13:15: @VirginTrains Yes. VT-231017-0019
4. **AGENT** `1558991` 2017-11-04 13:17: @481789 If you DM us we can have a look for you. ^PA

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `customer_service_complaint`, margin 0.0147)  
> Auto resolution: `redirected_to_dm`, resolved: False, DM redirect: True, outcome `redirected_to_dm`  
> Type: redirected_to_dm. Outcome: redirected_to_dm. Brand agent turns: 2. Signals seen: redirected_to_dm, clarification_requested. Evidence: [tweet 1558991, redirected_to_dm] "If you DM us we can have a look for you."

Fill in CSV row `case_1558990`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 051 · `case_994755`

**First customer message:** Our @VirginTrains pendolino on it's travels in the @43637 labs! #hellobrum @355630 https://t.co/fpH7J8zmW5

**Conversation**

1. **CUSTOMER** `994755` 2017-10-14 11:58: Our @VirginTrains pendolino on it's travels in the @43637 labs! #hellobrum @355630 https://t.co/fpH7J8zmW5
2. **AGENT** `994754` 2017-10-14 12:10: @355629 @43637 @355630 Nice! Have a great journey guys 😊 🚄 ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `praise_positive_feedback`, margin 0.0321)  
> Auto resolution: `information_provided`, resolved: True, DM redirect: False, outcome `agent_closed`  
> Type: information_provided. Outcome: agent_closed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 994754, information_provided] "Nice! Have a great journey guys 😊 🚄"

Fill in CSV row `case_994755`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 052 · `case_915005`

**First customer message:** @VirginTrains poor catering on the 7am KingsX to Edinburgh. Now the boilers broken ... so no service

**Conversation**

1. **CUSTOMER** `915005` 2017-10-21 08:44: @VirginTrains poor catering on the 7am KingsX to Edinburgh. Now the boilers broken ... so no service
2. **AGENT** `915004` 2017-10-21 08:45: @337283 Oh no, can @120576 assist please? ^MW

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `praise_positive_feedback`, margin 0.043)  
> Auto resolution: `other`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: other. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 915004, other] "Oh no, can @120576 assist please?"

Fill in CSV row `case_915005`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 053 · `case_1609711`

**First customer message:** @VirginTrains is it possible to use a booked train ticket on an earlier service?

**Conversation**

1. **CUSTOMER** `1609711` 2017-11-05 10:25: @VirginTrains is it possible to use a booked train ticket on an earlier service?
2. **AGENT** `1609710` 2017-11-05 10:34: @494186 I'm afraid not Sam no. ^PA

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `service_status_delay_enquiry`, margin 0.0635)  
> Auto resolution: `other`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: other. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 1609710, other] "I'm afraid not Sam no."

Fill in CSV row `case_1609711`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 054 · `case_1494890`

**First customer message:** Currently sat on the floor of the train stationary. I have got a bulging disc in my back and am in agony. Great work @VirginTrains

**Conversation**

1. **CUSTOMER** `1494890` 2017-12-01 17:29: Currently sat on the floor of the train stationary. I have got a bulging disc in my back and am in agony. Great work @VirginTrains
2. **AGENT** `1494889` 2017-12-01 17:31: @232965 Which service are you on today? ^MW
3. **CUSTOMER** `1494888` 2017-12-01 17:31: @VirginTrains Milton Keynes Central to Euston
4. **AGENT** `1494886` 2017-12-01 17:37: @232965 I see, have you spoken to the team onboard about this? ^MW
5. **CUSTOMER** `1494887` 2017-12-01 17:38: @VirginTrains I can’t move. Blocked in. I am at the far end of the train wedged in. There are people between the doors. People in vestibules and between seats.

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `praise_positive_feedback`, margin 0.0213)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: unresolved. Outcome: customer_followup_unanswered. Brand agent turns: 2. Signals seen: clarification_requested. Evidence: [tweet 1494886, clarification_requested] "I see, have you spoken to the team onboard about this?"

Fill in CSV row `case_1494890`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 055 · `case_1388536`

**First customer message:** Hi @VirginTrains I left my coat on the last 9:15am Birmingham to Glasgow yesterday. Is there a way to see if it's been found? X

**Conversation**

1. **CUSTOMER** `1388536` 2017-10-28 10:20: Hi @VirginTrains I left my coat on the last 9:15am Birmingham to Glasgow yesterday. Is there a way to see if it's been found? X
2. **AGENT** `1388535` 2017-10-28 10:22: @443036 Oh no! Please call our team on 03331 031 031 option 1, 3 ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `praise_positive_feedback`, margin 0.0096)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 1388535, information_provided] "Oh no! Please call our team on 03331 031 031 option 1, 3"

Fill in CSV row `case_1388536`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 056 · `case_1640404`

**First customer message:** @VirginTrains will ticket acceptance continue tomorrow (6th) for routes north on other providers?

**Conversation**

1. **CUSTOMER** `1640404` 2017-11-05 23:37: @VirginTrains will ticket acceptance continue tomorrow (6th) for routes north on other providers?
2. **AGENT** `1640403` 2017-11-05 23:38: @501513 Hi Isabel, where are you booked to travel to? ^KS
3. **CUSTOMER** `1640402` 2017-11-05 23:39: @VirginTrains Sunderland w/GC but could head to Ncl with Virgin if we’re able to?
4. **AGENT** `1640400` 2017-11-05 23:42: @501513 Who is your ticket booked for, Isabel? ^MM
5. **CUSTOMER** `1640401` 2017-11-05 23:44: @VirginTrains Grand central. We were told ticket acceptance was on but there were no trains north to use it on. Wondered if mutual acceptance tomm too?

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `service_status_delay_enquiry`, margin 0.1479)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: unresolved. Outcome: customer_followup_unanswered. Brand agent turns: 2. Signals seen: clarification_requested. Evidence: [tweet 1640400, clarification_requested] "Who is your ticket booked for, Isabel?"

Fill in CSV row `case_1640404`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 057 · `case_2546362`

**First customer message:** @VirginTrains can you change the toilet in use indicator from red and green on your toilet doors? Labels maybe? Red green colourblind here... 👀

**Conversation**

1. **CUSTOMER** `2546362` 2017-11-16 14:22: @VirginTrains can you change the toilet in use indicator from red and green on your toilet doors? Labels maybe? Red green colourblind here... 👀
2. **AGENT** `2546361` 2017-11-16 14:23: @723730 We'll pass your comments on to our Fleet team Carl ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `customer_service_complaint`, margin 0.0147)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: feedback_acknowledged. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: feedback_acknowledged. Evidence: [tweet 2546361, feedback_acknowledged] "We'll pass your comments on to our Fleet team Carl"

Fill in CSV row `case_2546362`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 058 · `case_2768813`

**First customer message:** @VirginTrains #berwickupontweed https://t.co/kKi9tARhwA

**Conversation**

1. **CUSTOMER** `2768813` 2017-11-21 12:04: @VirginTrains #berwickupontweed https://t.co/kKi9tARhwA
2. **AGENT** `2768812` 2017-11-21 12:04: @122190 Great snap! @120576 ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `unclear_or_media_only` (cluster -1 `unclear_or_media_only`; runner-up `unclear_or_media_only`, margin 0.0518)  
> Auto resolution: `other`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: other. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 2768812, other] "Great snap! @120576"

Fill in CSV row `case_2768813`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 059 · `case_2935587`

**First customer message:** @VirginTrains - appalling service on 9.56am Crewe to London... Over booked, crowded train - standing room only to London... Great start to the day!!! Booked seats already occupied...

**Conversation**

1. **CUSTOMER** `2935587` 2017-11-29 10:23: @VirginTrains - appalling service on 9.56am Crewe to London... Over booked, crowded train - standing room only to London... Great start to the day!!! Booked seats already occupied...
2. **AGENT** `2935584` 2017-11-29 10:28: @811615 Sorry to hear this Ian, did you have a reserved seat on this train? ^BT
3. **CUSTOMER** `2935586` 2017-11-29 10:31: @VirginTrains Yes, I did... But, lots of people occupying the wrong seats and passengers standing in every isle and doorway...
4. **CUSTOMER** `2935583` 2017-11-29 10:50: @VirginTrains You need a longer train - and better training for your conducting staff... You shouldn't be defending the fact this service is so poor - a bit of empathy and a simple apology would be much more appropriate... And effective...
5. **CUSTOMER** `2935585` 2017-11-29 10:52: @VirginTrains It might also be sensible and better customer management - not too mention safer!!! To allow standing passengers to move into the empty 1st class carriages...
6. **AGENT** `2935582` 2017-11-29 10:52: @811615 Sorry about the inconvenience Ian, if you wish to make a formal complaint regarding this you can do so here https://t.co/t20UbyOCSn ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 1 `service_running_enquiry`; runner-up `journey_disruption_complaint`, margin 0.0352)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: self_service. Outcome: redirected_to_channel. Brand agent turns: 2. Signals seen: self_service, clarification_requested. Evidence: [tweet 2935582, self_service] "Sorry about the inconvenience Ian, if you wish to make a formal complaint regarding this you can do so here https://t.co/t20UbyOCSn"

Fill in CSV row `case_2935587`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 060 · `case_1991647`

**First customer message:** The @VirginTrains WiFi is about as much fun as Branson turning up at Nobu and touching your lasses face eh @38512 eh? ...total fail!

**Conversation**

1. **CUSTOMER** `1991647` 2017-10-18 19:02: The @VirginTrains WiFi is about as much fun as Branson turning up at Nobu and touching your lasses face eh @38512 eh? ...total fail!
2. **AGENT** `1991646` 2017-10-18 19:03: @589428 @38512 What issues are you having? ^PA

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `onboard_wifi_issue` (cluster 11 `onboard_wifi_issue`; runner-up `chitchat_non_support`, margin 0.1231)  
> Auto resolution: `clarification_requested`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: clarification_requested. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: clarification_requested. Evidence: [tweet 1991646, clarification_requested] "What issues are you having?"

Fill in CSV row `case_1991647`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 061 · `case_2690471`

**First customer message:** On my way back from Edinburgh with @VirginTrains booked on the 12:30 and no seats allocated, trains looks twice overbooked, having to stand, really not impressed

**Conversation**

1. **CUSTOMER** `2690471` 2017-11-19 15:57: On my way back from Edinburgh with @VirginTrains booked on the 12:30 and no seats allocated, trains looks twice overbooked, having to stand, really not impressed
2. **AGENT** `2690468` 2017-11-19 16:06: @756376 Hi Thibault, it sounds like you're travelling with @120576 today so we'll pass this over for you ^HP
3. **CUSTOMER** `2690470` 2017-11-19 16:07: @VirginTrains @120576 Would at least expect a refund https://t.co/n7yAa8xpIx
4. **OTHER-AGENT** `2690469` 2017-11-19 16:09: @VirginTrains @756376 Really sorry to hear this - did you have a reserved seat at all? ^KM
5. **CUSTOMER** `2690472` 2017-11-19 16:15: @120576 @VirginTrains Booked the return via Trainline on 29 August, ticket showed seats on the way up, none on the way down despite booking specifically that return train. How can you possibly sell more tickets than actual seats on the train????
6. **OTHER-AGENT** `2690473` 2017-11-19 16:17: @756376 @VirginTrains We would advise speaking with @123241 as to why you weren't allocated seats ^KM
7. **CUSTOMER** `2690474` 2017-11-19 16:21: @120576 @VirginTrains @123241 Regardless of who I booked it through the fact is that @120576 is selling more tickets than seats on the train

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `journey_disruption_complaint`, margin 0.033)  
> Auto resolution: `redirected_to_other_operator`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: redirected_to_other_operator. Outcome: customer_followup_unanswered. Brand agent turns: 1. Signals seen: redirected_to_other_operator. Evidence: [tweet 2690468, redirected_to_other_operator] "Hi Thibault, it sounds like you're travelling with @120576 today so we'll pass this over for you"

Fill in CSV row `case_2690471`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 062 · `case_924810`

**First customer message:** @VirginTrains .. freezing cold on the MAN - EUS train .. all the way here .. the heating never came on .. carnt wait to get off

**Conversation**

1. **CUSTOMER** `924810` 2017-10-21 13:47: @VirginTrains .. freezing cold on the MAN - EUS train .. all the way here .. the heating never came on .. carnt wait to get off
2. **AGENT** `924808` 2017-10-21 13:51: @339473 Sorry to hear that, did you speak to staff regarding it? ^PA
3. **CUSTOMER** `924809` 2017-10-21 14:36: @VirginTrains Yes .. he said he would turn the heating up it never went up .. I get this train every week I swear I have never been so cold ..
4. **AGENT** `924811` 2017-10-21 14:37: @339473 Sorry for that. ^PA

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `service_status_delay_enquiry`, margin 0.0187)  
> Auto resolution: `clarification_requested`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: clarification_requested. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: clarification_requested, other. Evidence: [tweet 924808, clarification_requested] "Sorry to hear that, did you speak to staff regarding it?"

Fill in CSV row `case_924810`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 063 · `case_2351547`

**First customer message:** @91951 @VirginTrains USELESS

**Conversation**

1. **CUSTOMER** `2351547` 2017-11-13 17:50: @91951 @VirginTrains USELESS
2. **AGENT** `2351546` 2017-11-13 17:52: @679617 @91951 We can't see Chloe's tweet as her account is private. If you wish to get in touch do please DM ^CB https://t.co/hhfk9c3ylv

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `unclear_or_media_only` (cluster -1 `unclear_or_media_only`; runner-up `unclear_or_media_only`, margin 0.0518)  
> Auto resolution: `redirected_to_dm`, resolved: False, DM redirect: True, outcome `redirected_to_dm`  
> Type: redirected_to_dm. Outcome: redirected_to_dm. Brand agent turns: 1. Signals seen: redirected_to_dm, self_service. Evidence: [tweet 2351546, redirected_to_dm] "If you wish to get in touch do please DM ^CB https://t.co/hhfk9c3ylv"

Fill in CSV row `case_2351547`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 064 · `case_1328159`

**First customer message:** @VirginTrains @140497 @LondonMidland @120576 @122155 @121428 Can you confirm reasonable swap- reading to Crewe rather than my Euston to Crewe. I have paid to get to reading

**Conversation**

1. **CUSTOMER** `1328159` 2017-10-27 11:57: @VirginTrains @140497 @LondonMidland @120576 @122155 @121428 Can you confirm reasonable swap- reading to Crewe rather than my Euston to Crewe. I have paid to get to reading
2. **AGENT** `1328156` 2017-10-27 12:03: @138763 That route wouldn't be valid in these circumstance. For Crewe the best option would be to travel via Derby and Stoke ^CB
3. **CUSTOMER** `1328158` 2017-10-27 12:32: @VirginTrains Think you need to consider a new policy. You can't broadcast other options then have restrictions. I only had minutes to decide, I've gone..
4. **CUSTOMER** `1328157` 2017-10-27 14:41: @VirginTrains Cross Country were fine with my ticket. So far made it to Birmingham about an hour late already and still 2 to go.

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 9 `service_status_update_request`; runner-up `seat_reservation_issue`, margin 0.0182)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: information_provided. Outcome: customer_followup_unanswered. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 1328156, information_provided] "That route wouldn't be valid in these circumstance. For Crewe the best option would be to travel via Derby and Stoke"

Fill in CSV row `case_1328159`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 065 · `case_2134750`

**First customer message:** @VirginTrains do you have a policy on recruiting rude ticket inspectors? First class carriage and price, third class passenger experience

**Conversation**

1. **CUSTOMER** `2134750` 2017-11-08 17:26: @VirginTrains do you have a policy on recruiting rude ticket inspectors? First class carriage and price, third class passenger experience
2. **AGENT** `2134749` 2017-11-08 17:31: @628051 Not great to hear, Kieran :( What's happened? ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `ticket_booking_query`, margin 0.0259)  
> Auto resolution: `clarification_requested`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: clarification_requested. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: clarification_requested. Evidence: [tweet 2134749, clarification_requested] "Not great to hear, Kieran :( What's happened?"

Fill in CSV row `case_2134750`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 066 · `case_1280931`

**First customer message:** @VirginTrains just seen a staff member on their phone and on their phone was porn. Lovely to see that in first class.

**Conversation**

1. **CUSTOMER** `1280931` 2017-10-26 16:59: @VirginTrains just seen a staff member on their phone and on their phone was porn. Lovely to see that in first class.
2. **AGENT** `1280929` 2017-10-26 17:04: @419704 Which service are you travelling on, Laura? ^MM
3. **CUSTOMER** `1280930` 2017-10-26 18:07: @VirginTrains 15.24 Shrewsbury to Euston. They let someone through and held their phone up.
4. **AGENT** `1280932` 2017-10-26 18:23: @419704 We'll pass your comments on regarding this situation ^MM
5. **CUSTOMER** `1280933` 2017-10-26 18:29: @VirginTrains I got the staff member name if you want it...?
6. **AGENT** `1280934` 2017-10-26 18:41: @419704 Can you send us a DM? ^MM https://t.co/hhfk9c3ylv

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `first_class_catering_issue` (cluster 4 `first_class_service_issue`; runner-up `chitchat_non_support`, margin 0.0161)  
> Auto resolution: `redirected_to_dm`, resolved: False, DM redirect: True, outcome `redirected_to_dm`  
> Type: redirected_to_dm. Outcome: redirected_to_dm. Brand agent turns: 3. Signals seen: redirected_to_dm, self_service, feedback_acknowledged, clarification_requested. Evidence: [tweet 1280934, redirected_to_dm] "Can you send us a DM?"

Fill in CSV row `case_1280931`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 067 · `case_73982`

**First customer message:** @VirginTrains hope you’re giving refunds to 1st passengers on what is now a declassified 09:35 from Manchester Piccadilly to London Euston.

**Conversation**

1. **CUSTOMER** `73982` 2017-11-30 10:30: @VirginTrains hope you’re giving refunds to 1st passengers on what is now a declassified 09:35 from Manchester Piccadilly to London Euston.
2. **AGENT** `73981` 2017-11-30 10:47: @132556 Sorry for this today Isabelle, if you log this with our Customer Resolutions team here https://t.co/t20UbyOCSn they will look into this for you. ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `praise_positive_feedback`, margin 0.0007)  
> Auto resolution: `escalated`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: escalated. Outcome: redirected_to_channel. Brand agent turns: 1. Signals seen: escalated, self_service. Evidence: [tweet 73981, escalated] "Sorry for this today Isabelle, if you log this with our Customer Resolutions team here https://t.co/t20UbyOCSn they will look into this for you."

Fill in CSV row `case_73982`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 068 · `case_2106693`

**First customer message:** Bought internet on @virgintrains 30 mins ago, hasn’t worked once 🙈

**Conversation**

1. **CUSTOMER** `2106693` 2017-11-08 08:42: Bought internet on @virgintrains 30 mins ago, hasn’t worked once 🙈
2. **AGENT** `2106692` 2017-11-08 08:46: @621738 Are you struggling to connect or with speed? ^MW
3. **CUSTOMER** `2106691` 2017-11-08 08:48: @VirginTrains Struggling to connect, it lets me join but says there’s no internet available but let me pay for it 🤔
4. **AGENT** `2106690` 2017-11-08 08:53: @621738 Try entering https://t.co/1JEOSVPtGN into your browser, Jade ^MW
5. **CUSTOMER** `2106689` 2017-11-08 08:54: @VirginTrains Still not working https://t.co/3ZUJ54xHce
6. **AGENT** `2106688` 2017-11-08 08:56: @621738 Oh no, try speaking to the Wi-Fi team on 0330 088 1271 for assistance ^MW

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `onboard_wifi_issue` (cluster 11 `onboard_wifi_issue`; runner-up `delay_repay_refund_claim`, margin 0.0509)  
> Auto resolution: `troubleshooting`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: troubleshooting. Outcome: agent_answered_unconfirmed. Brand agent turns: 3. Signals seen: troubleshooting, self_service, information_provided, clarification_requested. Evidence: [tweet 2106690, troubleshooting] "Try entering https://t.co/1JEOSVPtGN into your browser, Jade"

Fill in CSV row `case_2106693`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 069 · `case_612277`

**First customer message:** Give the train man on the 12:15 from Leeds to Kings Cross speaking over the tannoy a pay rise please @VirginTrains 😂😂 he's hilarious!

**Conversation**

1. **CUSTOMER** `612277` 2017-11-22 13:21: Give the train man on the 12:15 from Leeds to Kings Cross speaking over the tannoy a pay rise please @VirginTrains 😂😂 he's hilarious!
2. **AGENT** `612275` 2017-11-22 13:24: @265605 Lovely one for @120576 ^LC
3. **OTHER-AGENT** `612276` 2017-11-22 13:27: @VirginTrains @265605 Thanks Hayley we'll pass on the praise! ^MS

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `praise_positive_feedback`, margin 0.0245)  
> Auto resolution: `other`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: other. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 612275, other] "Lovely one for @120576"

Fill in CSV row `case_612277`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 070 · `case_1276489`

**First customer message:** convenient that both my trains this weekend have been 25 mins late, because it’s only a 30 min delay you can refund @GWRHelp @VirginTrains

**Conversation**

1. **CUSTOMER** `1276489` 2017-10-16 07:37: convenient that both my trains this weekend have been 25 mins late, because it’s only a 30 min delay you can refund @GWRHelp @VirginTrains
2. **AGENT** `1276488` 2017-10-16 07:43: @352746 @GWRHelp What services where you travelling on, Elspeth? ^ BT
3. **CUSTOMER** `1276490` 2017-10-16 07:43: @GWRHelp @VirginTrains @nationalrailenq do you deliberately recommend trains to be 25 minutes delayed so you don’t have to refund/compensate people’s journeys?
4. **CUSTOMER** `1276486` 2017-10-16 07:45: @VirginTrains @GWRHelp Saturday morning Milton Keynes - Birmingham 09:13. I missed my connection to Redditch.
5. **OTHER-AGENT** `1286578` 2017-10-16 07:46: @352746 Hi there, we do not issue refunds and compensation I'm afraid.
6. **CUSTOMER** `1286577` 2017-10-16 07:46: @nationalrailenq I’m talking about the train services, that you as the overarching body oversee, do you not?
7. **OTHER-AGENT** `1286576` 2017-10-16 07:51: @352746 We answer train enquires and give advise in disruptions. We do not run train services, sell tickets or own stations.
8. **AGENT** `1276487` 2017-10-16 07:59: @352746 @GWRHelp 1/2 This would have meant your next service would have been the 10:42, giving an overall delay of 18 minutes. Which would
9. **AGENT** `1276485` 2017-10-16 08:02: @352746 @GWRHelp 2/2 mean no delay would be given unfortunately. ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `service_status_delay_enquiry`, margin 0.083)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 3. Signals seen: information_provided, clarification_requested. Evidence: [tweet 1276487, information_provided] "This would have meant your next service would have been the 10:42, giving an overall delay of 18 minutes. Which would"; [tweet 1276485, information_provided] "mean no delay would be given unfortunately."

Fill in CSV row `case_1276489`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 071 · `case_2881040`

**First customer message:** @116602 were going to make a documentary on rail commuters in the UK. After costing the price of tickets for the film crew, they decided to buy @6174 from @1561 &amp; commission another 10 series. It's cheaper. @VirginTrains @124107 @131370 @120586

**Conversation**

1. **CUSTOMER** `2881040` 2017-11-28 08:28: @116602 were going to make a documentary on rail commuters in the UK. After costing the price of tickets for the film crew, they decided to buy @6174 from @1561 &amp; commission another 10 series. It's cheaper. @VirginTrains @124107 @131370 @120586
2. **AGENT** `2881039` 2017-11-28 08:31: @799441 @124107 @131370 Anything we can help you with this morning? ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `ticket_booking_query`, margin 0.0895)  
> Auto resolution: `other`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: other. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 2881039, other] "Anything we can help you with this morning?"

Fill in CSV row `case_2881040`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 072 · `case_132651`

**First customer message:** #adulting forgot to swipe my @122550 when I was getting my ticket for @VirginTrains. I feel i've cheated out of life. #loyaltypoints #somanycards

**Conversation**

1. **CUSTOMER** `132651` 2017-11-24 07:59: #adulting forgot to swipe my @122550 when I was getting my ticket for @VirginTrains. I feel i've cheated out of life. #loyaltypoints #somanycards
2. **AGENT** `132649` 2017-11-24 08:01: @121933 @122550 Oh Sarah. Where's the restart button on Friday morning? ^CB
3. **CUSTOMER** `132650` 2017-11-24 08:08: @VirginTrains @122550 Should be in a coffee cup. Can I claim it back online or at a station?
4. **AGENT** `132652` 2017-11-24 08:14: @121933 @122550 It's not possible to add the points after purchase sadly ^CB
5. **CUSTOMER** `132653` 2017-11-24 08:16: @VirginTrains @122550 Noooo! https://t.co/l7z1WuUCAa
6. **AGENT** `132654` 2017-11-24 08:18: @121933 @122550 😕 ^CB https://t.co/yt2iG2kbh0

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `ticket_booking_query`, margin 0.001)  
> Auto resolution: `troubleshooting`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: troubleshooting. Outcome: redirected_to_channel. Brand agent turns: 3. Signals seen: troubleshooting, self_service, information_provided, clarification_requested. Evidence: [tweet 132649, troubleshooting] "Where's the restart button on Friday morning?"

Fill in CSV row `case_132651`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 073 · `case_1729294`

**First customer message:** Hey @VirginTrains why do letters to your Birmingham complaints dept come back "not known at this address"?

**Conversation**

1. **CUSTOMER** `1729294` 2017-10-17 18:58: Hey @VirginTrains why do letters to your Birmingham complaints dept come back "not known at this address"?
2. **AGENT** `1729292` 2017-10-17 18:59: @522679 Hi Caro, that's unusual, which address did you send the letter to please? ^HP
3. **CUSTOMER** `1729293` 2017-10-17 19:40: @VirginTrains This one - see pic https://t.co/mb8d6gzhuz
4. **AGENT** `1729295` 2017-10-17 19:51: @522679 Apologies Caro, we're not sure why this has been returned to you but we don have an online form you can submit if this helps?
5. **CUSTOMER** `1729296` 2017-10-17 21:02: @VirginTrains I'm tweeting for a relative who doesn't do the internet. So post is the only way for her to register a complaint.
6. **AGENT** `1729298` 2017-10-17 21:07: @522679 The only postal address we have is for a office address that we have since re-located from, we may have a divert in place ^JH
7. **AGENT** `1729297` 2017-10-17 21:08: @522679 "Freepost" (Customer Relations, Virgin Trains, FREEPOST BM6613, Meridian, 85 Smallbrook Queensway, BIRMINGHAM, B5 4HA) ^JH

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `delay_repay_refund_claim`, margin 0.1379)  
> Auto resolution: `escalated`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: escalated. Outcome: agent_answered_unconfirmed. Brand agent turns: 4. Signals seen: escalated, self_service, information_provided, clarification_requested. Evidence: [tweet 1729297, escalated] ""Freepost" (Customer Relations, Virgin Trains, FREEPOST BM6613, Meridian, 85 Smallbrook Queensway, BIRMINGHAM, B5 4HA)"

Fill in CSV row `case_1729294`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 074 · `case_50353`

**First customer message:** Sat here waiting for a train to go watch the reds and there's no entertainment at all..I blame @VirginTrains they should be giving us a brew

**Conversation**

1. **CUSTOMER** `50353` 2017-11-01 17:23: Sat here waiting for a train to go watch the reds and there's no entertainment at all..I blame @VirginTrains they should be giving us a brew
2. **AGENT** `50352` 2017-11-01 17:30: @127292 Which service are you on, Kristen? ^MM
3. **CUSTOMER** `50351` 2017-11-01 17:32: @VirginTrains Northern from Wigan to Liverpool lime st... worst thing is nowhere to charge my phone on here :-(
4. **AGENT** `50349` 2017-11-01 17:39: @127292 I'm afraid, Northern would be best to advise ^MM
5. **CUSTOMER** `50350` 2017-11-01 17:39: @VirginTrains Buy them out make them an offer they can't refuse

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `chitchat_non_support`, margin 0.0226)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: information_provided. Outcome: customer_followup_unanswered. Brand agent turns: 2. Signals seen: information_provided, clarification_requested. Evidence: [tweet 50349, information_provided] "I'm afraid, Northern would be best to advise"

Fill in CSV row `case_50353`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 075 · `case_513313`

**First customer message:** @VirginTrains why are trains sitting outside of wilmslow not moving?

**Conversation**

1. **CUSTOMER** `513313` 2017-12-01 23:23: @VirginTrains why are trains sitting outside of wilmslow not moving?
2. **AGENT** `513312` 2017-12-01 23:29: @238115 There's an issue at Manchester Benji. Apologies for the delays ^MW
3. **CUSTOMER** `513311` 2017-12-01 23:31: @VirginTrains When will the issues be fixed? It’s literally 100 yards from the platform. Just let people off.
4. **AGENT** `513309` 2017-12-01 23:39: @238115 Asap we hope but more will be announced when known ^MW
5. **CUSTOMER** `513310` 2017-12-01 23:43: @VirginTrains Why is it 8billion degrees on this fuking thing? By the time we wait for the dead bloke to be cleaned of the tracks in Manchester I will have melted

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `service_status_delay_enquiry`, margin 0.0263)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: information_provided. Outcome: customer_followup_unanswered. Brand agent turns: 2. Signals seen: information_provided. Evidence: [tweet 513312, information_provided] "There's an issue at Manchester Benji. Apologies for the delays"; [tweet 513309, information_provided] "Asap we hope but more will be announced when known"

Fill in CSV row `case_513313`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 076 · `case_1550709`

**First customer message:** @VirginTrains @213768 any idea what the delay is on the 19:17 train to stockport? Thanks

**Conversation**

1. **CUSTOMER** `1550709` 2017-11-19 18:57: @VirginTrains @213768 any idea what the delay is on the 19:17 train to stockport? Thanks
2. **AGENT** `1550708` 2017-11-19 19:34: @479935 @213768 Hi Chris, where are you travelling from please? ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 1 `service_running_enquiry`; runner-up `service_status_delay_enquiry`, margin 0.0572)  
> Auto resolution: `clarification_requested`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: clarification_requested. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: clarification_requested. Evidence: [tweet 1550708, clarification_requested] "Hi Chris, where are you travelling from please?"

Fill in CSV row `case_1550709`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 077 · `case_895330`

**First customer message:** @VirginTrains absolutely chaos at Euston!! Please talk to your staff so their messages are consistent. 19:00 to Manc. #OnOrOff

**Conversation**

1. **CUSTOMER** `895330` 2017-10-13 18:14: @VirginTrains absolutely chaos at Euston!! Please talk to your staff so their messages are consistent. 19:00 to Manc. #OnOrOff
2. **AGENT** `895327` 2017-10-13 18:23: @332586 Not great to hear, Matthew, what's happened? ^HP
3. **CUSTOMER** `895329` 2017-10-13 18:25: @VirginTrains Violent passenger on the train. @120893 struggled to remove. Train delayed. Over crowded. Passengers advised to board later service. Then...
4. **CUSTOMER** `895328` 2017-10-13 18:26: @VirginTrains Once said passengers left the train. Platform staff began shouting at passengers to get on the train. Chaos. Over crowding now in all areas
5. **CUSTOMER** `895325` 2017-10-13 18:26: @VirginTrains Finally. Tickets declassified. #WorthTheUpgrade #Not
6. **AGENT** `895322` 2017-10-13 18:29: @332586 I see, were you looking to make a complaint today, Matthew? ^MW
7. **CUSTOMER** `895323` 2017-10-13 18:30: @VirginTrains Just want people to be safe. We rely on your expertise to get us from A to B safely. Having passengers on top of each other. Scrambling to..
8. **CUSTOMER** `895324` 2017-10-13 18:31: @VirginTrains ..Get back on a train they were just told to get off. Is very unsafe. I own a model railway, yours is real, people are not toys or cargo.
9. **AGENT** `895326` 2017-10-13 18:34: @332586 We'll certainly pass your comments on about this, Matthew. Apologies for any inconvenience caused ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `praise_positive_feedback` (cluster 8 `praise_positive_feedback`; runner-up `customer_service_complaint`, margin 0.018)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: feedback_acknowledged. Outcome: agent_answered_unconfirmed. Brand agent turns: 3. Signals seen: feedback_acknowledged, clarification_requested, other. Evidence: [tweet 895326, feedback_acknowledged] "We'll certainly pass your comments on about this, Matthew."

Fill in CSV row `case_895330`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 078 · `case_1359376`

**First customer message:** Trying day for everyone at @VirginTrains today I’m sure. But staff on 19.23 from Euston have been brilliant - thank you! https://t.co/x74fYmUb57

**Conversation**

1. **CUSTOMER** `1359376` 2017-10-27 19:53: Trying day for everyone at @VirginTrains today I’m sure. But staff on 19.23 from Euston have been brilliant - thank you! https://t.co/x74fYmUb57
2. **AGENT** `1359375` 2017-10-27 20:02: @436926 Thank you for this feedback! If you wish to pass this on, you can do so here https://t.co/KLZij95Jry ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `praise_positive_feedback` (cluster 8 `praise_positive_feedback`; runner-up `service_status_delay_enquiry`, margin 0.0752)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: self_service. Outcome: redirected_to_channel. Brand agent turns: 1. Signals seen: self_service, feedback_acknowledged. Evidence: [tweet 1359375, self_service] "If you wish to pass this on, you can do so here https://t.co/KLZij95Jry"

Fill in CSV row `case_1359376`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 079 · `case_1382220`

**First customer message:** @VirginTrains Are you for f****** real! As farcical as the Ryan Air debacle!! 🤔😏 https://t.co/kAqHPl5lCO

**Conversation**

1. **CUSTOMER** `1382220` 2017-10-28 06:50: @VirginTrains Are you for f****** real! As farcical as the Ryan Air debacle!! 🤔😏 https://t.co/kAqHPl5lCO
2. **AGENT** `1382218` 2017-10-28 06:52: @441692 Unfortunately services are facing cancellations following major disruption on our network yesterday ^CB
3. **CUSTOMER** `1382219` 2017-10-28 06:58: @VirginTrains Nicely put! 😏. So will I be get a refund if you’re unable to keep to your side of the agreement and get me to B’Ham Intl?
4. **AGENT** `1382221` 2017-10-28 07:01: @441692 If you are delayed by over 30 mins as a result of issues today you'd be able to claim for this: https://t.co/0kAI0i2NBL ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `customer_service_complaint`, margin 0.1138)  
> Auto resolution: `compensation`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: compensation. Outcome: redirected_to_channel. Brand agent turns: 2. Signals seen: compensation, self_service, information_provided. Evidence: [tweet 1382221, compensation] "If you are delayed by over 30 mins as a result of issues today you'd be able to claim for this: https://t.co/0kAI0i2NBL"

Fill in CSV row `case_1382220`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 080 · `case_1352461`

**First customer message:** @VirginTrains can I get a refund for a train that is an hour delayed?

**Conversation**

1. **CUSTOMER** `1352461` 2017-10-27 17:36: @VirginTrains can I get a refund for a train that is an hour delayed?
2. **AGENT** `1352460` 2017-10-27 18:02: @435399 Hi Lola, you can, please claim Delay Repay here: https://t.co/fEE35w9nNF ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `journey_disruption_complaint`, margin 0.2087)  
> Auto resolution: `compensation`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: compensation. Outcome: redirected_to_channel. Brand agent turns: 1. Signals seen: compensation, self_service. Evidence: [tweet 1352460, compensation] "Hi Lola, you can, please claim Delay Repay here: https://t.co/fEE35w9nNF"

Fill in CSV row `case_1352461`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 081 · `case_2781354`

**First customer message:** @O2 literally every date I've selected in the promotional period comes up with this error?? What to do? https://t.co/9JSWEtZJ2t

**Conversation**

1. **CUSTOMER** `2781354` 2017-11-21 14:32: @O2 literally every date I've selected in the promotional period comes up with this error?? What to do? https://t.co/9JSWEtZJ2t
2. **CUSTOMER** `2781353` 2017-11-21 16:40: @O2 Umm can anyone at @VirginTrains help?
3. **AGENT** `2781352` 2017-11-21 16:43: @282020 @O2 What journey details are you putting in and when are you travelling, Paige? ^MW
4. **CUSTOMER** `2781351` 2017-11-21 16:45: @VirginTrains @O2 I was trying 5-7 Jan or 12-14 Jan, travelling London to Edinburgh
5. **AGENT** `2781350` 2017-11-21 16:49: @282020 @O2 Have you entered to travel from London Euston to Edinburgh ^MW
6. **OTHER-AGENT** `2781355` 2017-11-21 17:55: @282020 😞 We'll let Virgin take it from here, Paige. We hope it's sorted for you soon 👍

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `ticket_booking_query`, margin 0.0175)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: information_provided, clarification_requested. Evidence: [tweet 2781350, information_provided] "Have you entered to travel from London Euston to Edinburgh"

Fill in CSV row `case_2781354`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 082 · `case_983194`

**First customer message:** Familiar @VirginTrains experience. https://t.co/9PLwcf4aCP

**Conversation**

1. **CUSTOMER** `983194` 2017-10-22 14:35: Familiar @VirginTrains experience. https://t.co/9PLwcf4aCP
2. **AGENT** `983192` 2017-10-22 14:40: @353096 Hi Dawn, is that on @120576? if so our friends there can help. ^PA
3. **CUSTOMER** `983193` 2017-10-22 14:42: @VirginTrains @120576 They won't be able to conjure empty seats out of the air...

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `unclear_or_media_only` (cluster -1 `unclear_or_media_only`; runner-up `unclear_or_media_only`, margin 0.0518)  
> Auto resolution: `redirected_to_other_operator`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: redirected_to_other_operator. Outcome: customer_followup_unanswered. Brand agent turns: 1. Signals seen: redirected_to_other_operator. Evidence: [tweet 983192, redirected_to_other_operator] "if so our friends there can help."

Fill in CSV row `case_983194`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 083 · `case_2646431`

**First customer message:** Changing it up with a new rail company this evening - @VirginTrains &amp; not the usual @124107 🚂 #southofthewall From Vikings in Glasgow to London &amp; the Plastic Surgery meeting @20717 - a combo worthy of a hemsworth surely @746665?! 😍 @20718 #SCLF17 life https://t.co/P7UQIzNnNc

**Conversation**

1. **CUSTOMER** `2646431` 2017-11-30 19:01: Changing it up with a new rail company this evening - @VirginTrains &amp; not the usual @124107 🚂 #southofthewall From Vikings in Glasgow to London &amp; the Plastic Surgery meeting @20717 - a combo worthy of a hemsworth surely @746665?! 😍 @20718 #SCLF17 life https://t.co/P7UQIzNnNc
2. **AGENT** `2646430` 2017-11-30 19:03: @746664 @124107 @20717 @746665 @20718 Welcome onboard, Karen :D We hope you enjoy your journey! ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `journey_disruption_complaint`, margin 0.0027)  
> Auto resolution: `information_provided`, resolved: True, DM redirect: False, outcome `agent_closed`  
> Type: information_provided. Outcome: agent_closed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 2646430, information_provided] "Welcome onboard, Karen :D We hope you enjoy your journey!"

Fill in CSV row `case_2646431`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 084 · `case_1701391`

**First customer message:** @VirginTrains hi, I had an awful experience yesterday at Euston trying to get to Manchester - it took me 8 hours instead of 2

**Conversation**

1. **CUSTOMER** `1701391` 2017-11-06 21:11: @VirginTrains hi, I had an awful experience yesterday at Euston trying to get to Manchester - it took me 8 hours instead of 2
2. **CUSTOMER** `1701389` 2017-11-06 21:12: @VirginTrains I bought my ticket at a station. Please can you tell me how I can obtain a refund and apply for compensation. Thanks
3. **AGENT** `1701390` 2017-11-06 21:16: @515860 1/2 Hi Caz, apologies, we were experiencing major disruption across our network yesterday due to an incident involving
4. **AGENT** `1701387` 2017-11-06 21:16: @515860 2/2 emergency services. You can claim compensation via our online form here: https://t.co/5voPTD7Des ^HP
5. **CUSTOMER** `1701388` 2017-11-06 21:21: @VirginTrains I know, the service in the station was also terrible. I will use the link. Thanks

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 9 `service_status_update_request`; runner-up `service_status_delay_enquiry`, margin 0.0257)  
> Auto resolution: `compensation`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: compensation. Outcome: customer_followup_unanswered. Brand agent turns: 2. Signals seen: compensation, self_service, information_provided. Evidence: [tweet 1701387, compensation] "You can claim compensation via our online form here: https://t.co/5voPTD7Des"

Fill in CSV row `case_1701391`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 085 · `case_888906`

**First customer message:** On board @VirginTrains bound for London. #fastasyouwantdriver 👨🏻‍✈️🚝💨 https://t.co/hubKt0Fo9h

**Conversation**

1. **CUSTOMER** `888906` 2017-10-13 17:29: On board @VirginTrains bound for London. #fastasyouwantdriver 👨🏻‍✈️🚝💨 https://t.co/hubKt0Fo9h
2. **AGENT** `888905` 2017-10-13 17:35: @331040 Have a great journey, James! ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 9 `service_status_update_request`; runner-up `praise_positive_feedback`, margin 0.0356)  
> Auto resolution: `other`, resolved: True, DM redirect: False, outcome `agent_closed`  
> Type: other. Outcome: agent_closed. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 888905, other] "Have a great journey, James!"

Fill in CSV row `case_888906`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 086 · `case_659073`

**First customer message:** Worst train journey ever! @VirginTrains absolutely raging. Was meant to be home 9:20 and i’m still not in London.

**Conversation**

1. **CUSTOMER** `659073` 2017-11-22 23:33: Worst train journey ever! @VirginTrains absolutely raging. Was meant to be home 9:20 and i’m still not in London.
2. **AGENT** `659072` 2017-11-22 23:39: @276924 Sorry to hear Rebecca, we have had major disruption on our network today due to flooding. You can claim compensation for this journey here https://t.co/v2F1XYIfEP ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `service_status_delay_enquiry`, margin 0.0409)  
> Auto resolution: `compensation`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: compensation. Outcome: redirected_to_channel. Brand agent turns: 1. Signals seen: compensation, self_service. Evidence: [tweet 659072, compensation] "You can claim compensation for this journey here https://t.co/v2F1XYIfEP"

Fill in CSV row `case_659073`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 087 · `case_1898499`

**First customer message:** When you buy tickets six mths in #Advance, but circumstances outside your control mean ye can't use them. &gt;_&lt; @145773 @VirginTrains

**Conversation**

1. **CUSTOMER** `1898499` 2017-10-18 12:41: When you buy tickets six mths in #Advance, but circumstances outside your control mean ye can't use them. &gt;_&lt; @145773 @VirginTrains
2. **AGENT** `1898497` 2017-10-18 12:44: @304384 @145773 Oh no you can make a change of journey for them for a fee. ^PA
3. **CUSTOMER** `1898498` 2017-10-18 13:01: @VirginTrains @145773 1/2 As it happens, cheapest to buy a SVR for my outward leg so all good on getting home. However as my Adv...
4. **CUSTOMER** `1898500` 2017-10-18 13:02: @VirginTrains @145773 2/2 Tix were each under £10 there's no point trying to exchange them. They're all e-tickts too as happens! :-p
5. **AGENT** `1898501` 2017-10-18 13:09: @304384 @145773 The admin fee is set at £10 for now, but we can certainly pass on your feddback around this. ^KS
6. **CUSTOMER** `1898502` 2017-10-18 13:18: @VirginTrains @145773 That's why I norm choose (No refund) e-tkts for fares &lt;£10. No sense trying to change those with fees due! :-)

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `customer_service_complaint`, margin 0.196)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: feedback_acknowledged. Outcome: customer_followup_unanswered. Brand agent turns: 2. Signals seen: feedback_acknowledged, information_provided. Evidence: [tweet 1898501, feedback_acknowledged] "The admin fee is set at £10 for now, but we can certainly pass on your feddback around this."

Fill in CSV row `case_1898499`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 088 · `case_527360`

**First customer message:** @VirginTrains Is the 7:30 from London to Glasgow going to be cancelled? #BeHonest

**Conversation**

1. **CUSTOMER** `527360` 2017-12-02 07:27: @VirginTrains Is the 7:30 from London to Glasgow going to be cancelled? #BeHonest
2. **AGENT** `527358` 2017-12-02 07:39: @242065 This service is currently running and is awaiting departure from London Euston. ^BT
3. **CUSTOMER** `527359` 2017-12-02 07:40: @VirginTrains Thanks for the update #HappyGirl 👸🏽

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 9 `service_status_update_request`; runner-up `service_status_delay_enquiry`, margin 0.1061)  
> Auto resolution: `information_provided`, resolved: True, DM redirect: False, outcome `customer_confirmed`  
> Type: information_provided. Outcome: customer_confirmed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 527358, information_provided] "This service is currently running and is awaiting departure from London Euston."

Fill in CSV row `case_527360`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 089 · `case_2678204`

**First customer message:** @VirginTrains if I'm travelling LIV-EUS on a Sunday + want to pay for 1st Class upgrade on train can I use lounge at Lime St ?

**Conversation**

1. **CUSTOMER** `2678204` 2017-10-31 14:32: @VirginTrains if I'm travelling LIV-EUS on a Sunday + want to pay for 1st Class upgrade on train can I use lounge at Lime St ?
2. **AGENT** `2678202` 2017-10-31 14:34: @424827 You'd need to do this onboard, so sadly entry to the Lounge wouldn't be available ^CB
3. **CUSTOMER** `2678203` 2017-10-31 17:46: @VirginTrains I thought so, I won't bother then as it's not worth it.

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `first_class_catering_issue` (cluster 4 `first_class_service_issue`; runner-up `ticket_booking_query`, margin 0.0672)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: information_provided. Outcome: customer_followup_unanswered. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 2678202, information_provided] "You'd need to do this onboard, so sadly entry to the Lounge wouldn't be available"

Fill in CSV row `case_2678204`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 090 · `case_2984277`

**First customer message:** @VirginTrains dear virgin trains west coast isn’t about time your first and standard class got updated looks little out off date

**Conversation**

1. **CUSTOMER** `2984277` 2017-11-20 19:13: @VirginTrains dear virgin trains west coast isn’t about time your first and standard class got updated looks little out off date
2. **AGENT** `2984276` 2017-11-20 19:26: @353530 Thanks for your feedback, Henry. We'll pass this onto our makeover department ^MM https://t.co/ThLnSQn8O3

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `first_class_catering_issue` (cluster 4 `first_class_service_issue`; runner-up `journey_disruption_complaint`, margin 0.0223)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: self_service. Outcome: redirected_to_channel. Brand agent turns: 1. Signals seen: self_service, feedback_acknowledged. Evidence: [tweet 2984276, self_service] "We'll pass this onto our makeover department ^MM https://t.co/ThLnSQn8O3"

Fill in CSV row `case_2984277`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 091 · `case_1220391`

**First customer message:** @VirginTrains very unhappy with the service I have just had from your call centre. Neer to correct a ticket error and they won't help!

**Conversation**

1. **CUSTOMER** `1220391` 2017-10-25 14:57: @VirginTrains very unhappy with the service I have just had from your call centre. Neer to correct a ticket error and they won't help!
2. **AGENT** `1220389` 2017-10-25 15:03: @406331 Hi Neil, really sorry to hear that. Can we ask what you need to correct with your ticket please? ^HP
3. **CUSTOMER** `1220390` 2017-10-25 15:07: @VirginTrains I had to change the return journey but when I checked confirmation this afternoon the ticket was the wrong date and in past!
4. **CUSTOMER** `1220392` 2017-10-25 15:12: @VirginTrains how can your own system allow a return ticket to be changed to a date before the outbound!
5. **AGENT** `1220393` 2017-10-25 15:15: @406331 That's really unusual Neil, can we take your booking reference number so we can have a look at your ticket please? ^HP
6. **CUSTOMER** `1220394` 2017-10-25 15:20: @VirginTrains 2337752126 changed return journey online given new reference 2338124627 this ticket is wrong and should be for next Tuesday.
7. **AGENT** `1220395` 2017-10-25 15:29: @406331 The new reference shows a ticket for the 18:56 Edinburgh to Wolverhampton on Tuesday 31st October - is this incorrect? ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `customer_service_complaint`, margin 0.0205)  
> Auto resolution: `clarification_requested`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: clarification_requested. Outcome: agent_awaiting_customer. Brand agent turns: 3. Signals seen: clarification_requested, other. Evidence: [tweet 1220389, clarification_requested] "Can we ask what you need to correct with your ticket please?"

Fill in CSV row `case_1220391`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 092 · `case_1333784`

**First customer message:** @VirginTrains I'm due out of London (to Liverpool) on the 16.07 - ballet tickets at 19.30 (v close to station) Any idea what my chances are?

**Conversation**

1. **CUSTOMER** `1333784` 2017-10-27 13:21: @VirginTrains I'm due out of London (to Liverpool) on the 16.07 - ballet tickets at 19.30 (v close to station) Any idea what my chances are?
2. **AGENT** `1333783` 2017-10-27 13:43: @431154 Hi Jemma, we're expecting residual delays today but we'd advise keeping an eye on our Live Updates here: https://t.co/lLbFt1Oi6t

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 1 `service_running_enquiry`; runner-up `ticket_booking_query`, margin 0.008)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: self_service. Outcome: redirected_to_channel. Brand agent turns: 1. Signals seen: self_service. Evidence: [tweet 1333783, self_service] "Hi Jemma, we're expecting residual delays today but we'd advise keeping an eye on our Live Updates here: https://t.co/lLbFt1Oi6t"

Fill in CSV row `case_1333784`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 093 · `case_1317879`

**First customer message:** Worst diet ever @VirginTrains https://t.co/5or1KmEu2C

**Conversation**

1. **CUSTOMER** `1317879` 2017-12-01 09:43: Worst diet ever @VirginTrains https://t.co/5or1KmEu2C
2. **AGENT** `1317878` 2017-12-01 09:45: @427819 We would agree! ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `first_class_catering_issue`, margin 0.0892)  
> Auto resolution: `other`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: other. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 1317878, other] "We would agree!"

Fill in CSV row `case_1317879`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 094 · `case_1264133`

**First customer message:** @VirginTrains hi, am I allowed to travel on an earlier train with my advance-reserved seat ticket? Thank u. 😀

**Conversation**

1. **CUSTOMER** `1264133` 2017-10-26 11:50: @VirginTrains hi, am I allowed to travel on an earlier train with my advance-reserved seat ticket? Thank u. 😀
2. **AGENT** `1264134` 2017-10-26 11:51: @416028 Afraid not, you would need to travel on the booked service only ^MW
3. **CUSTOMER** `1264132` 2017-10-26 11:58: @VirginTrains Are there any charges then to swap my ticket for another train before departure?thanks
4. **AGENT** `1264131` 2017-10-26 11:59: @416028 It will depend on the ticket really, if you speak to Aftersales on 0344 556 5650 they can assist further ^MW

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `service_status_delay_enquiry`, margin 0.023)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: information_provided. Evidence: [tweet 1264134, information_provided] "Afraid not, you would need to travel on the booked service only"; [tweet 1264131, information_provided] "It will depend on the ticket really, if you speak to Aftersales on 0344 556 5650 they can assist further"

Fill in CSV row `case_1264133`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 095 · `case_1110614`

**First customer message:** @VirginTrains can you advise on progress for refund VT-130917-0277? Promised refund on 25/9, still waiting + asked another Q, no reply since

**Conversation**

1. **CUSTOMER** `1110614` 2017-10-15 10:12: @VirginTrains can you advise on progress for refund VT-130917-0277? Promised refund on 25/9, still waiting + asked another Q, no reply since
2. **AGENT** `1110615` 2017-10-15 10:18: @321139 1/2 Hi Graham, I'll escalate this with our Customer Resolutions team for you, as it looks like they're yet to respond to your
3. **AGENT** `1110613` 2017-10-15 10:18: @321139 2/2 latest emails, so you should hear back from them within the next few days :) ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `customer_service_complaint`, margin 0.1724)  
> Auto resolution: `escalated`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: escalated. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: escalated, information_provided. Evidence: [tweet 1110615, escalated] "Hi Graham, I'll escalate this with our Customer Resolutions team for you, as it looks like they're yet to respond to your"

Fill in CSV row `case_1110614`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 096 · `case_2721235`

**First customer message:** @VirginTrains I️ have a split journey and 1st train running late will advance ticket be valid on next journey?

**Conversation**

1. **CUSTOMER** `2721235` 2017-11-20 10:10: @VirginTrains I️ have a split journey and 1st train running late will advance ticket be valid on next journey?
2. **AGENT** `2721234` 2017-11-20 10:12: @763516 It will be, do you have a through ticket, Simone? ^MW

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 1 `service_running_enquiry`; runner-up `ticket_booking_query`, margin 0.0034)  
> Auto resolution: `clarification_requested`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: clarification_requested. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: clarification_requested. Evidence: [tweet 2721234, clarification_requested] "It will be, do you have a through ticket, Simone?"

Fill in CSV row `case_2721235`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 097 · `case_604609`

**First customer message:** @VirginTrains It would be v handy if the app could update status when tix have been collected. I tried to collect tix from the station on Monday but had to pay £83 for a new ticket as I forgot I'd already collected them way back in July. I'd like a refund for the unused ones.

**Conversation**

1. **CUSTOMER** `604609` 2017-11-22 10:26: @VirginTrains It would be v handy if the app could update status when tix have been collected. I tried to collect tix from the station on Monday but had to pay £83 for a new ticket as I forgot I'd already collected them way back in July. I'd like a refund for the unused ones.
2. **AGENT** `604608` 2017-11-22 10:28: @263722 I'm not able to authorise a refund from here I'm afraid. You'll need to get in touch with our team through the website: https://t.co/t20Ubz6dJV ^CB
3. **CUSTOMER** `604607` 2017-11-22 10:33: @VirginTrains I know you're not. I was making a suggestion. Sadly some Virgin Trains staff are so unhelpful &amp; seemed to enjoy watching me have a breakdown. I'm quite sure there'll be a reason for a refund refusal but I'd like you to forward the suggestion as it would solve the problem.
4. **AGENT** `604606` 2017-11-22 10:34: @263722 I'm not in a position to do this sadly, and under these circumstances it is likely that a refund wouldn't be processed. Do please get in touch so that the team can take a further look ^CB
5. **CUSTOMER** `604605` 2017-11-22 10:38: @VirginTrains I will but it's time Virgin had compassion. I think it's disgusting to charge so much money &amp; offer no refund when presented with evidence that I have a set of unused tickets - same person, same journey. Where do I suggest the app change?
6. **AGENT** `604604` 2017-11-22 10:40: @263722 You can write in via https://t.co/eBwvq4sf1W ^PA

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `ticket_booking_query`, margin 0.0015)  
> Auto resolution: `refund`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: refund. Outcome: redirected_to_channel. Brand agent turns: 3. Signals seen: refund, self_service. Evidence: [tweet 604608, refund] "I'm not able to authorise a refund from here I'm afraid."; [tweet 604606, refund] "I'm not in a position to do this sadly, and under these circumstances it is likely that a refund wouldn't be processed."

Fill in CSV row `case_604609`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 098 · `case_1651339`

**First customer message:** 2/2 that can be done! Cool @VirginTrains

**Conversation**

1. **CUSTOMER** `1651339` 2017-11-06 06:41: 2/2 that can be done! Cool @VirginTrains
2. **AGENT** `1651337` 2017-11-06 06:45: @504312 Hi, Gemma, staff at the ticket office may be able to assist you. ^PH
3. **CUSTOMER** `1651338` 2017-11-06 06:47: @VirginTrains Afraid not, he said — and I quote “there’s nothing I can do, you need to call them “

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `first_class_catering_issue`, margin 0.0779)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: self_service. Outcome: customer_followup_unanswered. Brand agent turns: 1. Signals seen: self_service. Evidence: [tweet 1651337, self_service] "Hi, Gemma, staff at the ticket office may be able to assist you."

Fill in CSV row `case_1651339`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 099 · `case_674836`

**First customer message:** @VirginTrains just been very well looked after on the 9.49 Durham to Birmingham by Malcolm on tickets and Trevor on the refreshment trolley - excellent and chirpy service! Thanks boys 💕

**Conversation**

1. **CUSTOMER** `674836` 2017-11-23 10:24: @VirginTrains just been very well looked after on the 9.49 Durham to Birmingham by Malcolm on tickets and Trevor on the refreshment trolley - excellent and chirpy service! Thanks boys 💕
2. **AGENT** `674834` 2017-11-23 10:25: @280948 Lovely one for @121428 :) ^LC
3. **CUSTOMER** `674835` 2017-11-23 10:28: @VirginTrains @121428 Indeed. Whole team can be proud! On time, clean and comfy and superb customer service with good humour.
4. **OTHER-AGENT** `674837` 2017-11-23 10:54: @280948 @VirginTrains Great to hear Karenza, we'll be sure to pass your comments on to Trevor and Malcolm :)

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `praise_positive_feedback` (cluster 8 `praise_positive_feedback`; runner-up `journey_disruption_complaint`, margin 0.1213)  
> Auto resolution: `other`, resolved: True, DM redirect: False, outcome `agent_closed`  
> Type: other. Outcome: agent_closed. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 674834, other] "Lovely one for @121428 :)"

Fill in CSV row `case_674836`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 100 · `case_1328972`

**First customer message:** @VirginTrains - bought a ticket. Can’t see it on the app. Was told to buy another ticket and get a refund. Ridiculous!! Make it easy.

**Conversation**

1. **CUSTOMER** `1328972` 2017-10-27 12:13: @VirginTrains - bought a ticket. Can’t see it on the app. Was told to buy another ticket and get a refund. Ridiculous!! Make it easy.
2. **AGENT** `1328971` 2017-10-27 12:19: @430070 Sorry to hear you've had problems with your ticket. Please get in touch so we can look in to this: https://t.co/6CQvcjqVwG ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `ticket_booking_query`, margin 0.0022)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: self_service. Outcome: redirected_to_channel. Brand agent turns: 1. Signals seen: self_service. Evidence: [tweet 1328971, self_service] "Please get in touch so we can look in to this: https://t.co/6CQvcjqVwG"

Fill in CSV row `case_1328972`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 101 · `case_2793943`

**First customer message:** @VirginTrains on hold for 30mins rn anyone home?

**Conversation**

1. **CUSTOMER** `2793943` 2017-11-21 21:19: @VirginTrains on hold for 30mins rn anyone home?
2. **AGENT** `2793942` 2017-11-21 21:19: @779697 Which number are you calling Kev? ^BT
3. **CUSTOMER** `2793941` 2017-11-21 21:23: @VirginTrains 03445565650
4. **AGENT** `2793939` 2017-11-21 21:25: @779697 We don't have a timescale for this at the moment. They are open until 22:00. ^BT
5. **CUSTOMER** `2793940` 2017-11-21 21:25: @VirginTrains 😭
6. **CUSTOMER** `2793938` 2017-11-21 21:30: @VirginTrains Can you help???
7. **AGENT** `2793936` 2017-11-21 21:34: @779697 What's happened, Kev? ^MW
8. **CUSTOMER** `2793937` 2017-11-21 21:39: @VirginTrains Printed my tickets for my journey tomorrow and they are both in the same name is this a problem?
9. **AGENT** `2793947` 2017-11-21 21:40: @779697 Are they e-tickets by any chance? ^MW
10. **CUSTOMER** `2793935` 2017-11-21 21:42: @VirginTrains The tickets are for 2 people
11. **AGENT** `2793934` 2017-11-21 21:43: @779697 Can we see a pic of them please? ^MW

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 2 `delay_in_progress_report`; runner-up `chitchat_non_support`, margin 0.0151)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: information_provided. Outcome: agent_awaiting_customer. Brand agent turns: 5. Signals seen: information_provided, clarification_requested, other. Evidence: [tweet 2793939, information_provided] "We don't have a timescale for this at the moment. They are open until 22:00."

Fill in CSV row `case_2793943`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 102 · `case_1387434`

**First customer message:** @VirginTrains @327380 It's probably staged by Corbyn

**Conversation**

1. **CUSTOMER** `1387434` 2017-10-28 11:21: @VirginTrains @327380 It's probably staged by Corbyn

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `customer_service_complaint`, margin 0.1129)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `no_response`  
> Brand never replied in the public thread (outcome: no_response).

Fill in CSV row `case_1387434`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 103 · `case_1210947`

**First customer message:** Delightful and helpful @VirginTrains manager on 9:40 GLA-&gt;LON this morning! Didn't catch his name but credit where credit due!

**Conversation**

1. **CUSTOMER** `1210947` 2017-10-25 11:42: Delightful and helpful @VirginTrains manager on 9:40 GLA-&gt;LON this morning! Didn't catch his name but credit where credit due!
2. **AGENT** `1210946` 2017-10-25 11:48: @404238 Great to hear! We will be sure to pass this onto him. ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `chitchat_non_support`, margin 0.0132)  
> Auto resolution: `information_provided`, resolved: True, DM redirect: False, outcome `agent_closed`  
> Type: information_provided. Outcome: agent_closed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 1210946, information_provided] "Great to hear! We will be sure to pass this onto him."

Fill in CSV row `case_1210947`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 104 · `case_1578716`

**First customer message:** @486184 Oh dear 😐

**Conversation**

1. **CUSTOMER** `1578716` 2017-11-04 18:25: @486184 Oh dear 😐

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `unclear_or_media_only` (cluster -1 `unclear_or_media_only`; runner-up `unclear_or_media_only`, margin 0.0518)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `no_response`  
> Brand never replied in the public thread (outcome: no_response).

Fill in CSV row `case_1578716`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 105 · `case_551597`

**First customer message:** @248556 The last day of my radiotherapy (after my cancer op) I had to get on the train home.The toilets STANK as usual on @VirginTrains I complained 2 staff(I explained I had just finished radiotherapy) on the platform abt it they treated me abominably &amp; did not let me get on the train.

**Conversation**

1. **CUSTOMER** `551597` 2017-12-02 18:16: @248556 The last day of my radiotherapy (after my cancer op) I had to get on the train home.The toilets STANK as usual on @VirginTrains I complained 2 staff(I explained I had just finished radiotherapy) on the platform abt it they treated me abominably &amp; did not let me get on the train.
2. **AGENT** `551595` 2017-12-02 19:06: @248555 @248556 Were you looking to make a complaint today, Michelle? ^MW
3. **CUSTOMER** `551596` 2017-12-02 23:25: @VirginTrains @248556 I made this complaint a long time ago on Twitter u did not bother to look into it.U employ vile ppl to deal with the public.#Branson's businesses r a disgrace including his trains.I expect his medical clinics r exactly the same #Virgin #revolting
4. **AGENT** `551598` 2017-12-02 23:56: @248555 Complaints should be made to this link &gt; https://t.co/UtCX17d8HR ^OL

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `seat_reservation_issue`, margin 0.0233)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: self_service. Outcome: redirected_to_channel. Brand agent turns: 2. Signals seen: self_service, other. Evidence: [tweet 551598, self_service] "Complaints should be made to this link > https://t.co/UtCX17d8HR"

Fill in CSV row `case_551597`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 106 · `case_139198`

**First customer message:** So @VirginTrains doors close 2mins before departure but only give 4 mins for passengers to get there - inc elderly 🤔

**Conversation**

1. **CUSTOMER** `139198` 2017-11-24 10:58: So @VirginTrains doors close 2mins before departure but only give 4 mins for passengers to get there - inc elderly 🤔
2. **AGENT** `139196` 2017-11-24 10:59: @147369 Hi Lyanne, which station are you referring to please? ^HP
3. **CUSTOMER** `139197` 2017-11-24 12:05: @VirginTrains Euston
4. **AGENT** `139199` 2017-11-24 12:07: @147369 Thanks for confirming, Lyanne. This has now changed to 1 minute before departure, but we'll pass your comments on in regards to this ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `service_status_delay_enquiry`, margin 0.0022)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: feedback_acknowledged. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: feedback_acknowledged, clarification_requested. Evidence: [tweet 139199, feedback_acknowledged] "This has now changed to 1 minute before departure, but we'll pass your comments on in regards to this"

Fill in CSV row `case_139198`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 107 · `case_1271248`

**First customer message:** @VirginTrains 0717 . Coach f why are there no unreserved seats on this service as per the norm

**Conversation**

1. **CUSTOMER** `1271248` 2017-10-16 06:16: @VirginTrains 0717 . Coach f why are there no unreserved seats on this service as per the norm
2. **AGENT** `1271247` 2017-10-16 06:21: @417630 Morning Jordan. Where are you travelling from and too? ^BT
3. **CUSTOMER** `1271246` 2017-10-16 06:21: @VirginTrains Preston to crewe .. normally coach f is free but it's more or less reserved and has been recently
4. **AGENT** `1271245` 2017-10-16 06:28: @417630 Coach C would be the new unreserved Coach on this service. ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `service_status_delay_enquiry`, margin 0.202)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: information_provided, clarification_requested. Evidence: [tweet 1271245, information_provided] "Coach C would be the new unreserved Coach on this service."

Fill in CSV row `case_1271248`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 108 · `case_2452993`

**First customer message:** @VirginTrains £88 I payed delayed over 1hour this morning and now on the way home I have to sit on the floor by the door because no seats???

**Conversation**

1. **CUSTOMER** `2452993` 2017-10-29 18:15: @VirginTrains £88 I payed delayed over 1hour this morning and now on the way home I have to sit on the floor by the door because no seats???
2. **AGENT** `2452990` 2017-10-29 18:17: @702792 Sorry to hear that, we do recommend pre-booking seats to avoid having to stand, apologies Matty ^MW
3. **CUSTOMER** `2452991` 2017-10-29 18:20: @VirginTrains How can u book seats if you don’t know what time ur coming back.the first class section is empty how about less first class I’m sat on floor
4. **AGENT** `2453005` 2017-10-29 18:22: @702792 We'll be sure to pass this on for you, Matty ^MW
5. **CUSTOMER** `2452992` 2017-10-29 18:22: @VirginTrains £88 and I sit on the floor is that right????
6. **CUSTOMER** `2452982` 2017-10-29 18:24: @VirginTrains Delayed an hour from Liverpool this morning and sit on the floor on the way home do you think this is acceptable???
7. **AGENT** `2452980` 2017-10-29 18:28: @702792 That's not good to hear, would you like to make a complaint? ^MW
8. **CUSTOMER** `2452981` 2017-10-29 18:31: @VirginTrains Yes of course theirs 4 of us who have spent £88 each
9. **AGENT** `2452983` 2017-10-29 18:32: @702792 Please submit your comments here - https://t.co/EFf9ENO7yO ^MW
10. **CUSTOMER** `2452988` 2017-10-29 18:34: @VirginTrains No I’d rather do it public because this is an absolute disgrace
11. **CUSTOMER** `2452987` 2017-10-29 18:35: @VirginTrains First class is never full but you put so many first class carriages on absolute disgrace
12. **CUSTOMER** `2452986` 2017-10-29 18:36: @VirginTrains £88 and I don’t get a seat do you think that is right
13. **CUSTOMER** `2452985` 2017-10-29 18:36: @VirginTrains That’s what you get for paying £88 https://t.co/O8D1Z2tHPB
14. **CUSTOMER** `2452984` 2017-10-29 18:37: @VirginTrains Sat on the floor
15. **CUSTOMER** `2452989` 2017-10-29 18:38: @VirginTrains Can you reply please

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `delay_repay_refund_claim`, margin 0.0224)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: self_service. Outcome: customer_followup_unanswered. Brand agent turns: 4. Signals seen: self_service, feedback_acknowledged, information_provided, other. Evidence: [tweet 2452983, self_service] "Please submit your comments here - https://t.co/EFf9ENO7yO"

Fill in CSV row `case_2452993`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 109 · `case_131881`

**First customer message:** The the tall guy who looks like Kylo Ren, who goes on the 07:35 @VirginTrains from Manchester to Stoke every morning, I think I love you lol

**Conversation**

1. **CUSTOMER** `131881` 2017-11-24 07:34: The the tall guy who looks like Kylo Ren, who goes on the 07:35 @VirginTrains from Manchester to Stoke every morning, I think I love you lol
2. **AGENT** `131880` 2017-11-24 07:36: @145733 It must be love, love, love! ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `praise_positive_feedback`, margin 0.0112)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 131880, information_provided] "It must be love, love, love!"

Fill in CSV row `case_131881`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 110 · `case_527314`

**First customer message:** @VirginTrains I need to get back to Preston from Euston around 10am, are trains still delayed?

**Conversation**

1. **CUSTOMER** `527314` 2017-12-02 07:37: @VirginTrains I need to get back to Preston from Euston around 10am, are trains still delayed?
2. **AGENT** `527313` 2017-12-02 07:58: @242055 Service are meeting with delay due to damage to the overhead wires between Euston and Watford Junction. We advise to keep up to date here https://t.co/mPukV7bSIe ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 1 `service_running_enquiry`; runner-up `service_status_delay_enquiry`, margin 0.1627)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: self_service. Outcome: redirected_to_channel. Brand agent turns: 1. Signals seen: self_service. Evidence: [tweet 527313, self_service] "We advise to keep up to date here https://t.co/mPukV7bSIe"

Fill in CSV row `case_527314`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 111 · `case_582377`

**First customer message:** Terrible end to our honeymoon @VirginTrains managed to get on the right service following the Euston delays to find we can’t have our 1st class seats we paid for because someone else is sat in them. Been travelling for 36 hours 😭😭😡😡

**Conversation**

1. **CUSTOMER** `582377` 2017-12-03 13:23: Terrible end to our honeymoon @VirginTrains managed to get on the right service following the Euston delays to find we can’t have our 1st class seats we paid for because someone else is sat in them. Been travelling for 36 hours 😭😭😡😡
2. **AGENT** `582375` 2017-12-03 13:27: @257291 We're so sorry to hear this, Mark. Unfortunately we've experienced major disruption across our network due to emergency repairs on overhead lines today. You can claim compensation for this by contacting our Customer Resolutions team here: https://t.co/LQOZhhqzcV
3. **CUSTOMER** `582376` 2017-12-03 13:31: @VirginTrains Will do- this is us currently stood up next to the toilet. 1st class all the way. Felt like more could have been done to protect the reservations. Appreciate the severe disruptions- just don’t think the customer experience has been handled particularly well. https://t.co/3qjWqBo7Ru
4. **CUSTOMER** `582378` 2017-12-03 13:34: @VirginTrains My wife is already furiously completing it. Thanks.
5. **AGENT** `582379` 2017-12-03 13:54: @257291 Apologies again for the inconvenience caused to you and your wife today ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `first_class_catering_issue`, margin 0.0348)  
> Auto resolution: `compensation`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: compensation. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: compensation, self_service, information_provided. Evidence: [tweet 582375, compensation] "You can claim compensation for this by contacting our Customer Resolutions team here: https://t.co/LQOZhhqzcV"

Fill in CSV row `case_582377`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 112 · `case_2978009`

**First customer message:** @VirginTrains abusing disabled people again. Stuck on train in Euston. #abuse #Disability #yetagain

**Conversation**

1. **CUSTOMER** `2978009` 2017-10-31 20:19: @VirginTrains abusing disabled people again. Stuck on train in Euston. #abuse #Disability #yetagain
2. **AGENT** `2978008` 2017-10-31 20:21: @821317 Which service were you on? Have the onboard team not been through? ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `customer_service_complaint`, margin 0.0543)  
> Auto resolution: `clarification_requested`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: clarification_requested. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: clarification_requested. Evidence: [tweet 2978008, clarification_requested] "Which service were you on?"

Fill in CSV row `case_2978009`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 113 · `case_786708`

**First customer message:** Thanks to the lady @VirginTrains help desk. Almost missed my train after queuingx15min to book space for my bike #BetterBikesOnTrainsService

**Conversation**

1. **CUSTOMER** `786708` 2017-10-12 09:31: Thanks to the lady @VirginTrains help desk. Almost missed my train after queuingx15min to book space for my bike #BetterBikesOnTrainsService
2. **AGENT** `786707` 2017-10-12 09:33: @307672 Hi there, do feel free to drop us a message next time and we can reserve a bike space for you here if there's availability :) ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `service_status_delay_enquiry`, margin 0.0178)  
> Auto resolution: `redirected_to_dm`, resolved: False, DM redirect: True, outcome `agent_answered_unconfirmed`  
> Type: redirected_to_dm. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: redirected_to_dm. Evidence: [tweet 786707, redirected_to_dm] "Hi there, do feel free to drop us a message next time and we can reserve a bike space for you here if there's availability :)"

Fill in CSV row `case_786708`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 114 · `case_1449670`

**First customer message:** Disappointed with @VirginTrains for having a faulty app, as a result costing me over £100 in replacement tickets. Not impressed.

**Conversation**

1. **CUSTOMER** `1449670` 2017-11-02 16:46: Disappointed with @VirginTrains for having a faulty app, as a result costing me over £100 in replacement tickets. Not impressed.
2. **AGENT** `1449669` 2017-11-02 16:48: @456403 Have you contacted the Aftersales support team about this issue? They can be reached on 03445565650 ^CB
3. **CUSTOMER** `1449668` 2017-11-02 16:54: @VirginTrains With zero success. Customer services are the reason I ended up spending the money. The after sales team offered to take more money from me.
4. **AGENT** `1449666` 2017-11-02 16:56: @456403 If you're unhappy with your experience and wish to escalate further please get in touch: https://t.co/wsh5ezzX7n ^CB
5. **CUSTOMER** `1449667` 2017-11-02 17:08: @VirginTrains Thank you.

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `customer_service_complaint`, margin 0.0315)  
> Auto resolution: `escalated`, resolved: True, DM redirect: False, outcome `customer_confirmed`  
> Type: escalated. Outcome: customer_confirmed. Brand agent turns: 2. Signals seen: escalated, self_service, clarification_requested. Evidence: [tweet 1449666, escalated] "If you're unhappy with your experience and wish to escalate further please get in touch: https://t.co/wsh5ezzX7n"

Fill in CSV row `case_1449670`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 115 · `case_2418514`

**First customer message:** What a shambles of a journey that was @VirginTrains between Southampton and Reading Late and severe over crowding

**Conversation**

1. **CUSTOMER** `2418514` 2017-10-29 11:32: What a shambles of a journey that was @VirginTrains between Southampton and Reading Late and severe over crowding
2. **AGENT** `2418513` 2017-10-29 11:35: @694901 We don't run that service, Richard. Sounds like one for @121428 ^CB
3. **CUSTOMER** `2418516` 2017-10-29 11:44: @VirginTrains Confused bought tickets on Virgin Train website and the train had Virgin on the side
4. **AGENT** `2418789` 2017-10-29 11:47: @694901 We don't run between Southampton and Reading, but our website will sell tickets for all other train operators ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `praise_positive_feedback` (cluster 8 `praise_positive_feedback`; runner-up `journey_disruption_complaint`, margin 0.008)  
> Auto resolution: `redirected_to_other_operator`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: redirected_to_other_operator. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: redirected_to_other_operator, self_service. Evidence: [tweet 2418513, redirected_to_other_operator] "We don't run that service, Richard."

Fill in CSV row `case_2418514`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 116 · `case_2685484`

**First customer message:** So when will @VirginTrains accommodate customers who pay for seats on the 43minutes past the hour from London&gt;Edinburgh. #LackOfPlanning

**Conversation**

1. **CUSTOMER** `2685484` 2017-10-31 16:59: So when will @VirginTrains accommodate customers who pay for seats on the 43minutes past the hour from London&gt;Edinburgh. #LackOfPlanning
2. **AGENT** `2685483` 2017-10-31 17:04: @755246 Which service are you referring to, Laxmi? ^LC
3. **CUSTOMER** `2685482` 2017-10-31 17:05: @VirginTrains the 4.43
4. **AGENT** `2685480` 2017-10-31 17:08: @755246 Have you reserved a seat prior to travelling? ^LC
5. **CUSTOMER** `2685477` 2017-10-31 17:09: @VirginTrains no but there are others I am standing with that have. Although it's not the first time. I travel regularly and this...
6. **CUSTOMER** `2685481` 2017-10-31 17:10: @VirginTrains particular service is always heavily booked! Must be at least 150 people standing
7. **AGENT** `2685475` 2017-10-31 17:10: @755246 We'd always advise reserving a seat prior to travelling as services into Scotland can get very busy I'm afraid, Laxmi ^LC
8. **CUSTOMER** `2685476` 2017-10-31 17:12: @VirginTrains I'm sure you can understand that most people book open returns. Just makes sense to have an additional service!
9. **AGENT** `2685478` 2017-10-31 17:14: @755246 It's not always possible to run additional trains sadly. We'll certainly pass on your comments to our timetable team ^CB
10. **CUSTOMER** `2685479` 2017-10-31 17:18: @VirginTrains yes please do considering a train manager advised an additional service would be ideal previously too!

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `service_status_delay_enquiry`, margin 0.1076)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: feedback_acknowledged. Outcome: customer_followup_unanswered. Brand agent turns: 4. Signals seen: feedback_acknowledged, information_provided, clarification_requested. Evidence: [tweet 2685478, feedback_acknowledged] "We'll certainly pass on your comments to our timetable team"

Fill in CSV row `case_2685484`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 117 · `case_1991622`

**First customer message:** Makes me sad when plug sockets don't work on @VirginTrains like I trusted you and you hurt my heart and left me with no battery.

**Conversation**

1. **CUSTOMER** `1991622` 2017-10-18 19:00: Makes me sad when plug sockets don't work on @VirginTrains like I trusted you and you hurt my heart and left me with no battery.
2. **AGENT** `1991619` 2017-10-18 19:02: @448165 If you speak to staff they can reset the power. ^PA
3. **CUSTOMER** `1991620` 2017-10-18 19:04: @VirginTrains I'd love to but I'm visually impaired and me finding the staff on here would be like you looking for your pal in a pitch black room 😀
4. **AGENT** `1991623` 2017-10-18 19:07: @448165 Oh right, could you try someone else's socket to see if they're having the same issue please? ^MW
5. **CUSTOMER** `1991624` 2017-10-18 19:08: @VirginTrains Yes they are, non working.
6. **AGENT** `1991625` 2017-10-18 19:09: @448165 Which service and coach are you in? ^PA
7. **CUSTOMER** `1991626` 2017-10-18 19:11: @VirginTrains Think I'm in G. On the 7.30pm from Birmingham New Street to London Euston
8. **AGENT** `1991627` 2017-10-18 19:17: @448165 We will try to contact the Train Manager for you. ^PA
9. **CUSTOMER** `1991621` 2017-10-18 19:19: @VirginTrains thank you. I think I can trust you again now 😉
10. **AGENT** `1991628` 2017-10-18 19:20: @448165 I've just spoken to them there is no coach G, are you able to see which is it so I can advise. ^PA
11. **CUSTOMER** `1991630` 2017-10-18 19:22: @VirginTrains Told you I was visually Impaired ha ha E is what I meant. Sorry.
12. **AGENT** `1991632` 2017-10-18 19:25: @448165 No problem. ^PA
13. **CUSTOMER** `1991629` 2017-10-18 19:25: @VirginTrains All working now. Top customer service and help. Cheers
14. **AGENT** `1991631` 2017-10-18 19:25: @448165 It should all be sorted now. ^PA

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `chitchat_non_support`, margin 0.0214)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 7. Signals seen: information_provided, clarification_requested, other. Evidence: [tweet 1991619, information_provided] "If you speak to staff they can reset the power."; [tweet 1991627, information_provided] "We will try to contact the Train Manager for you."; [tweet 1991628, information_provided] "I've just spoken to them there is no coach G, are you able to see which is it so I can advise."; [tweet 1991631, information_provided] "It should all be sorted now."

Fill in CSV row `case_1991622`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 118 · `case_1201667`

**First customer message:** @VirginTrains no one checked my ticket from Warrington to London on Saturday at 10:20. Can I get a refund for this? Pointless ticket

**Conversation**

1. **CUSTOMER** `1201667` 2017-10-25 13:00: @VirginTrains no one checked my ticket from Warrington to London on Saturday at 10:20. Can I get a refund for this? Pointless ticket
2. **AGENT** `1201665` 2017-10-25 13:05: @402345 Hi Mike, I'm afraid we wouldn't be able to offer a refund for this. You are still required to have a ticket for your journey ^HP
3. **CUSTOMER** `1201666` 2017-10-25 15:22: @VirginTrains I know I'm just being awkward. Just annoys that some people would have got that journey for free! Why did no one check it?
4. **AGENT** `1201669` 2017-10-25 15:30: @402345 1/2 We can totally understand your frustration, Mike, apologies about this. We're not sure why this was the case but we'll be sure
5. **AGENT** `1201668` 2017-10-25 15:30: @402345 2/2 to pass your comments on in regards to this issue ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `delay_repay_refund_claim`, margin 0.0197)  
> Auto resolution: `refund`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: refund. Outcome: agent_answered_unconfirmed. Brand agent turns: 3. Signals seen: refund, feedback_acknowledged, information_provided. Evidence: [tweet 1201665, refund] "Hi Mike, I'm afraid we wouldn't be able to offer a refund for this."

Fill in CSV row `case_1201667`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 119 · `case_1438856`

**First customer message:** @VirginTrains need help with a booking

**Conversation**

1. **CUSTOMER** `1438856` 2017-11-02 10:48: @VirginTrains need help with a booking
2. **AGENT** `1438855` 2017-11-02 10:54: @453919 You will need to contact @120576 regarding this. ^PA

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `seat_reservation_issue`, margin 0.0265)  
> Auto resolution: `redirected_to_other_operator`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: redirected_to_other_operator. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: redirected_to_other_operator. Evidence: [tweet 1438855, redirected_to_other_operator] "You will need to contact @120576 regarding this."

Fill in CSV row `case_1438856`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 120 · `case_690906`

**First customer message:** @VirginTrains #virgintrains Really pleased I paid for a 1st class ticket to Glasgow tonight. Carriage was filthy and no service in over 3 hours. https://t.co/UiMV9K7RoD

**Conversation**

1. **CUSTOMER** `690906` 2017-11-19 23:36: @VirginTrains #virgintrains Really pleased I paid for a 1st class ticket to Glasgow tonight. Carriage was filthy and no service in over 3 hours. https://t.co/UiMV9K7RoD
2. **AGENT** `690905` 2017-11-19 23:44: @285089 Stuart, please provide full details via https://t.co/bqqiy8tv2g &amp; we'll follow up for you ^AN

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `first_class_catering_issue` (cluster 4 `first_class_service_issue`; runner-up `ticket_booking_query`, margin 0.0488)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: self_service. Outcome: redirected_to_channel. Brand agent turns: 1. Signals seen: self_service. Evidence: [tweet 690905, self_service] "Stuart, please provide full details via https://t.co/bqqiy8tv2g & we'll follow up for you"

Fill in CSV row `case_690906`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 121 · `case_688682`

**First customer message:** @VirginTrains Currently waiting at Lancaster for the 13:55 to Glasgow. Board just says delayed and been no announcement on how delayed it will be. Any idea?

**Conversation**

1. **CUSTOMER** `688682` 2017-11-23 14:11: @VirginTrains Currently waiting at Lancaster for the 13:55 to Glasgow. Board just says delayed and been no announcement on how delayed it will be. Any idea?
2. **AGENT** `688681` 2017-11-23 14:13: @284530 We're currently investigating this delay ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 2 `delay_in_progress_report`; runner-up `service_status_delay_enquiry`, margin 0.0544)  
> Auto resolution: `other`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: other. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 688681, other] "We're currently investigating this delay"

Fill in CSV row `case_688682`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 122 · `case_803431`

**First customer message:** Paid £5 for wi-fi that doesn't work. Fantastic, thank you so much @VirginTrains

**Conversation**

1. **CUSTOMER** `803431` 2017-10-12 16:23: Paid £5 for wi-fi that doesn't work. Fantastic, thank you so much @VirginTrains
2. **AGENT** `803429` 2017-10-12 16:34: @311464 What issues are you having? ^PA
3. **CUSTOMER** `803430` 2017-10-12 16:46: @VirginTrains Was unable to so much as load a single page in first two hours. Is now just painfully slow. Really poor for a paying service.
4. **AGENT** `803432` 2017-10-12 16:49: @311464 Sorry to hear that Brett, are you still onboard? ^PA
5. **CUSTOMER** `803433` 2017-10-12 16:50: @VirginTrains I am yes, it's the 2pm Glasgow Central to London Euston service.
6. **AGENT** `803434` 2017-10-12 16:53: @311464 Try closing it all down now and resetting it with https://t.co/1JEOSVPtGN ^PA

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `onboard_wifi_issue` (cluster 11 `onboard_wifi_issue`; runner-up `customer_service_complaint`, margin 0.3153)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: self_service. Outcome: redirected_to_channel. Brand agent turns: 3. Signals seen: self_service, clarification_requested. Evidence: [tweet 803434, self_service] "Try closing it all down now and resetting it with https://t.co/1JEOSVPtGN"

Fill in CSV row `case_803431`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 123 · `case_1403228`

**First customer message:** @VirginTrains what’s the point in making a reservation 4 the 19.05 KGX when u declare all reservations cancelled

**Conversation**

1. **CUSTOMER** `1403228` 2017-10-28 17:54: @VirginTrains what’s the point in making a reservation 4 the 19.05 KGX when u declare all reservations cancelled
2. **AGENT** `1403224` 2017-10-28 17:55: @446253 Could @120576 assist please? ^MW
3. **OTHER-AGENT** `1403225` 2017-10-28 18:00: @VirginTrains @446253 Apologies Sarah, we're using a different train-type this evening and it's affected the planned reservations. ^JC
4. **CUSTOMER** `1403227` 2017-10-28 18:02: @VirginTrains @120576 Having 2 sit in a ‘free 4 all’ with passengers demanding 2 sit in their pre booked seats, but having 2 explain no seat reservations
5. **CUSTOMER** `1403226` 2017-10-28 18:03: @VirginTrains @120576 I can see as I am on it. So are many irritated folk demanding their seats. Like the lady who just shouted at us. Thanks for that!

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `ticket_booking_query`, margin 0.0452)  
> Auto resolution: `other`, resolved: True, DM redirect: False, outcome `customer_confirmed`  
> Type: other. Outcome: customer_confirmed. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 1403224, other] "Could @120576 assist please?"

Fill in CSV row `case_1403228`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 124 · `case_1119691`

**First customer message:** @VirginTrains this is how we stood for nearly 3 hrs after being delayed 3 hrs and turfed off put on another train! Not a replacement! #crazy

**Conversation**

1. **CUSTOMER** `1119691` 2017-10-15 12:47: @VirginTrains this is how we stood for nearly 3 hrs after being delayed 3 hrs and turfed off put on another train! Not a replacement! #crazy
2. **AGENT** `1119689` 2017-10-15 12:53: @384170 Hi Ria, sorry to hear this, which service were you travelling on please? ^HP
3. **CUSTOMER** `1119690` 2017-10-15 15:37: @VirginTrains The 7.55 Newcastle to London. 3 hrs delayed then put on an already packed train #awful
4. **AGENT** `1119692` 2017-10-15 15:42: @384170 Thanks for confirming Ria. We'll pass this over to our colleagues at @120576 as they'd be best to advise on this ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `service_status_delay_enquiry`, margin 0.0912)  
> Auto resolution: `redirected_to_other_operator`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: redirected_to_other_operator. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: redirected_to_other_operator, clarification_requested. Evidence: [tweet 1119692, redirected_to_other_operator] "We'll pass this over to our colleagues at @120576 as they'd be best to advise on this"

Fill in CSV row `case_1119691`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 125 · `case_2012137`

**First customer message:** Very disappointed with @VirginTrains Customer Service. Receive an email to say refund will be issued within 5-7 days. 2 weeks later and still no sign and no response to any of my emails #expectedbetter

**Conversation**

1. **CUSTOMER** `2012137` 2017-12-01 12:09: Very disappointed with @VirginTrains Customer Service. Receive an email to say refund will be issued within 5-7 days. 2 weeks later and still no sign and no response to any of my emails #expectedbetter
2. **AGENT** `2012134` 2017-12-01 12:18: @594979 Do you have a VT reference you could DM us please? ^LC
3. **CUSTOMER** `2012135` 2017-12-01 12:21: @VirginTrains Will you need to follow me so I can DM?
4. **CUSTOMER** `2012136` 2017-12-01 12:28: @VirginTrains Thank you - just DM’d the VT number…

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `customer_service_complaint`, margin 0.0833)  
> Auto resolution: `redirected_to_dm`, resolved: True, DM redirect: True, outcome `customer_confirmed`  
> Type: redirected_to_dm. Outcome: customer_confirmed. Brand agent turns: 1. Signals seen: redirected_to_dm, clarification_requested. Evidence: [tweet 2012134, redirected_to_dm] "Do you have a VT reference you could DM us please?"

Fill in CSV row `case_2012137`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 126 · `case_1332161`

**First customer message:** @VirginTrains my mum is stuck on the 11.15 Manchester to Euston train but stuck at Stafford - are there any updates on when it'll be moving?

**Conversation**

1. **CUSTOMER** `1332161` 2017-10-27 12:55: @VirginTrains my mum is stuck on the 11.15 Manchester to Euston train but stuck at Stafford - are there any updates on when it'll be moving?
2. **AGENT** `1332160` 2017-10-27 13:10: @430774 Services are facing major disruption due to a person being hit by a train. We're waiting on further updates ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 1 `service_running_enquiry`; runner-up `service_status_delay_enquiry`, margin 0.1032)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 1332160, information_provided] "Services are facing major disruption due to a person being hit by a train. We're waiting on further updates"

Fill in CSV row `case_1332161`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 127 · `case_624278`

**First customer message:** @VirginTrains how do I apply for my 50% refund? I've got a Virgin only year season ticket for my daily commute Wilmslow to Piccadilly

**Conversation**

1. **CUSTOMER** `624278` 2017-11-22 17:56: @VirginTrains how do I apply for my 50% refund? I've got a Virgin only year season ticket for my daily commute Wilmslow to Piccadilly
2. **AGENT** `624277` 2017-11-22 18:02: @268190 Hi Ray, you can still apply for this by contacting our Customer Resolutions team here: https://t.co/t20UbyOCSn ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `ticket_booking_query`, margin 0.0542)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: self_service. Outcome: redirected_to_channel. Brand agent turns: 1. Signals seen: self_service. Evidence: [tweet 624277, self_service] "Hi Ray, you can still apply for this by contacting our Customer Resolutions team here: https://t.co/t20UbyOCSn"

Fill in CSV row `case_624278`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 128 · `case_2785867`

**First customer message:** The conductor just came on to ask 2nd class passengers not to exit via the 1st class please... Aww bless, poor sensibilities. It's standing room only here, @VirginTrains 1706 KX-Leeds.

**Conversation**

1. **CUSTOMER** `2785867` 2017-11-21 18:10: The conductor just came on to ask 2nd class passengers not to exit via the 1st class please... Aww bless, poor sensibilities. It's standing room only here, @VirginTrains 1706 KX-Leeds.
2. **AGENT** `2785865` 2017-11-21 18:13: @777801 We'll tag @120576 in this for you, Charlie ^MW
3. **OTHER-AGENT** `2785866` 2017-11-21 18:15: @VirginTrains @777801 Hi Charlie, have you been unable to get a seat on your journey tonight? ^JC

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `first_class_catering_issue`, margin 0.0327)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: information_provided. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 2785865, information_provided] "We'll tag @120576 in this for you, Charlie"

Fill in CSV row `case_2785867`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 129 · `case_1406904`

**First customer message:** @VirginTrains help lost my card that I’ve used to book a ticket with!

**Conversation**

1. **CUSTOMER** `1406904` 2017-10-28 18:58: @VirginTrains help lost my card that I’ve used to book a ticket with!
2. **AGENT** `1406903` 2017-10-28 19:00: @447012 Can you provide the booking reference via DM and we can look into this for you? ^BT https://t.co/hhfk9c3ylv

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `delay_repay_refund_claim`, margin 0.1129)  
> Auto resolution: `redirected_to_dm`, resolved: False, DM redirect: True, outcome `redirected_to_dm`  
> Type: redirected_to_dm. Outcome: redirected_to_dm. Brand agent turns: 1. Signals seen: redirected_to_dm, escalated, self_service, clarification_requested. Evidence: [tweet 1406903, redirected_to_dm] "Can you provide the booking reference via DM and we can look into this for you?"

Fill in CSV row `case_1406904`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 130 · `case_1686854`

**First customer message:** @127472 @VirginTrains what's up with your customer service, 1:30 on hold and no one willing to help with a simple request!

**Conversation**

1. **CUSTOMER** `1686854` 2017-10-17 16:23: @127472 @VirginTrains what's up with your customer service, 1:30 on hold and no one willing to help with a simple request!
2. **AGENT** `1686853` 2017-10-17 16:31: @512298 @127472 Hi Joe, which number have you contacted please? ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `service_status_delay_enquiry`, margin 0.0797)  
> Auto resolution: `clarification_requested`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: clarification_requested. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: clarification_requested. Evidence: [tweet 1686853, clarification_requested] "Hi Joe, which number have you contacted please?"

Fill in CSV row `case_1686854`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 131 · `case_1201655`

**First customer message:** @VirginTrains Beam is being a bollocks again! Half way through a film and off! Tried all the usual and not working, 1330 Eus-Glasgow

**Conversation**

1. **CUSTOMER** `1201655` 2017-10-25 13:09: @VirginTrains Beam is being a bollocks again! Half way through a film and off! Tried all the usual and not working, 1330 Eus-Glasgow
2. **AGENT** `1201653` 2017-10-25 13:13: @402342 Oh no, sorry to hear that Stuart, are you receiving a specific error message at all? ^HP
3. **CUSTOMER** `1201654` 2017-10-25 13:15: @VirginTrains Nope, just says please connect to WiFi and try again. I’m in coach E was perfect then nowt!
4. **AGENT** `1201656` 2017-10-25 13:22: @402342 Have you tried forgetting the network in your settings, reconnecting &amp; following https://t.co/1JEOSVPtGN ^HP
5. **CUSTOMER** `1201657` 2017-10-25 13:26: @VirginTrains Tried it all, had the same issue a few weeks ago. No worries I’m off at Warrington so will have a Kip instead!
6. **AGENT** `1201658` 2017-10-25 13:30: @402342 Really sorry about that, Stuart :( We hope this isn't the case next time you travel. Enjoy your nap though! 💤 ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `customer_service_complaint`, margin 0.0032)  
> Auto resolution: `troubleshooting`, resolved: True, DM redirect: False, outcome `agent_closed`  
> Type: troubleshooting. Outcome: agent_closed. Brand agent turns: 3. Signals seen: troubleshooting, self_service, information_provided, clarification_requested. Evidence: [tweet 1201653, troubleshooting] "Oh no, sorry to hear that Stuart, are you receiving a specific error message at all?"; [tweet 1201656, troubleshooting] "Have you tried forgetting the network in your settings, reconnecting & following https://t.co/1JEOSVPtGN"

Fill in CSV row `case_1201655`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 132 · `case_2675219`

**First customer message:** best joke I've heard all week is @VirginTrains charging people £3.80 for a sandwich

**Conversation**

1. **CUSTOMER** `2675219` 2017-11-19 12:36: best joke I've heard all week is @VirginTrains charging people £3.80 for a sandwich
2. **AGENT** `2675217` 2017-11-19 12:38: @753040 They're nice sarnies though, Adz! ^MW
3. **CUSTOMER** `2675218` 2017-11-19 12:40: @VirginTrains I wish I could try it but you bent me over so much already with the cost of a ticket if I bought the sandwich my back might break

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `customer_service_complaint`, margin 0.038)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: unresolved. Outcome: customer_followup_unanswered. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 2675217, other] "They're nice sarnies though, Adz!"

Fill in CSV row `case_2675219`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 133 · `case_703523`

**First customer message:** @VirginTrains @127472 I made a genuine error in booking a train ticket 24hrs later than I should have and you charged me £36.50 extra for it, it’s a crap way to treat your customers who already pay high levels to travel. Doncaster to KX has cost me £140 https://t.co/0U5s0bTdKl

**Conversation**

1. **CUSTOMER** `703523` 2017-11-23 17:44: @VirginTrains @127472 I made a genuine error in booking a train ticket 24hrs later than I should have and you charged me £36.50 extra for it, it’s a crap way to treat your customers who already pay high levels to travel. Doncaster to KX has cost me £140 https://t.co/0U5s0bTdKl
2. **AGENT** `703521` 2017-11-23 17:47: @288068 @127472 Sorry to see this Ruchard, looks as though you are travelling with @120576 today though. ^BT
3. **CUSTOMER** `703522` 2017-11-23 17:48: @VirginTrains @127472 @120576 Oh that’s ok then
4. **OTHER-AGENT** `703524` 2017-11-23 17:49: @288068 @VirginTrains @127472 Hi Ruchard, can you DM me with the original booking please? ^JC

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `journey_disruption_complaint`, margin 0.0039)  
> Auto resolution: `redirected_to_other_operator`, resolved: False, DM redirect: False, outcome `redirected_to_dm`  
> Type: redirected_to_other_operator. Outcome: redirected_to_dm. Brand agent turns: 1. Signals seen: redirected_to_other_operator. Evidence: [tweet 703521, redirected_to_other_operator] "Sorry to see this Ruchard, looks as though you are travelling with @120576 today though."

Fill in CSV row `case_703523`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 134 · `case_27574`

**First customer message:** @VirginTrains On worst Train journey in long time from stockport to euston 9.43 total overcrowding and no declassification #trainmonopolies

**Conversation**

1. **CUSTOMER** `27574` 2017-11-01 10:20: @VirginTrains On worst Train journey in long time from stockport to euston 9.43 total overcrowding and no declassification #trainmonopolies
2. **AGENT** `27571` 2017-11-01 10:23: @122032 Sorry to hear that Sarah, have you spoken with any staff onboard? ^PA
3. **CUSTOMER** `27573` 2017-11-01 10:27: @VirginTrains Yes and the Manager response to everyone is to leave the Train and get next. We have business meetings as soon as arrive so can’t
4. **AGENT** `27578` 2017-11-01 10:30: @122032 I see, sorry it's so busy for you, had you reserved a seat? ^PA
5. **CUSTOMER** `27572` 2017-11-01 10:39: @VirginTrains Poor show Train Manager is hiding in first class and won’t come and speak to us
6. **AGENT** `27575` 2017-11-01 10:44: @122032 Please accept our apologies Sarah. But it would be the train manager who would be best to speak to regarding this. ^BT
7. **CUSTOMER** `27576` 2017-11-01 11:03: @VirginTrains We would if didn’t hide #poorcustomerservice #shouldhave declassified
8. **AGENT** `27577` 2017-11-01 11:05: @122032 I'll pass on your feedback. ^PA

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `service_status_delay_enquiry`, margin 0.0571)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: feedback_acknowledged. Outcome: agent_answered_unconfirmed. Brand agent turns: 4. Signals seen: feedback_acknowledged, information_provided, clarification_requested, other. Evidence: [tweet 27577, feedback_acknowledged] "I'll pass on your feedback."

Fill in CSV row `case_27574`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 135 · `case_2835069`

**First customer message:** @VirginTrains hey guys. Did you get my DM, not heard back? If you did please can you respond to it :-)

**Conversation**

1. **CUSTOMER** `2835069` 2017-11-27 14:40: @VirginTrains hey guys. Did you get my DM, not heard back? If you did please can you respond to it :-)
2. **AGENT** `2835067` 2017-11-27 14:41: @789155 Just refresh it or send another please? ^MW
3. **CUSTOMER** `2835068` 2017-11-27 14:42: @VirginTrains Will resend it :-)

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `chitchat_non_support`, margin 0.0312)  
> Auto resolution: `troubleshooting`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: troubleshooting. Outcome: customer_followup_unanswered. Brand agent turns: 1. Signals seen: troubleshooting. Evidence: [tweet 2835067, troubleshooting] "Just refresh it or send another please?"

Fill in CSV row `case_2835069`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 136 · `case_1236520`

**First customer message:** @VirginTrains our rain was delayed+ is now running so slowly it's taken a hour to reach Milton Keynes, there's no card machine in the (1/3)

**Conversation**

1. **CUSTOMER** `1236520` 2017-10-25 18:47: @VirginTrains our rain was delayed+ is now running so slowly it's taken a hour to reach Milton Keynes, there's no card machine in the (1/3)
2. **CUSTOMER** `1236519` 2017-10-25 18:48: @VirginTrains Shop, so I'll be spending the rest of the evening without any food or drink. The reason for this being trains were all cancelled &amp; (2/3)
3. **CUSTOMER** `1236518` 2017-10-25 18:49: @VirginTrains Then suddenly back on, so there was no time to grab anything. Having an absolutely vile evening. (3/3) @410011
4. **CUSTOMER** `1236513` 2017-10-25 18:50: @VirginTrains @410011 As per usual my virgin trains experience has been completely horrific.
5. **AGENT** `1236517` 2017-10-25 18:52: @410010 @410011 1/2 We apologies for the inconvenience caused, Liv. Unfortunately we are experiencing major disruption across our
6. **AGENT** `1236511` 2017-10-25 18:53: @410010 @410011 2/2 network due to a fatality on our network earlier this evening ^HP
7. **CUSTOMER** `1236512` 2017-10-25 18:54: @VirginTrains @410011 I don't see how that effects the card machine on my deathly slow train?
8. **CUSTOMER** `1236514` 2017-10-25 18:54: @VirginTrains @410011 So now I'm late and starving and dehydrated. Late is unavoidable. The lack on working facilities is.
9. **AGENT** `1236516` 2017-10-25 19:06: @410010 @410011 1/2 Apologies Liv, I was responding to your first tweet about your train being delayed &amp; slow running
10. **AGENT** `1236515` 2017-10-25 19:06: @410010 @410011 2/2 We'll pass your comments on in regards to the card machine. Apologies for the inconvenience ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 2 `delay_in_progress_report`; runner-up `customer_service_complaint`, margin 0.1009)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: feedback_acknowledged. Outcome: agent_answered_unconfirmed. Brand agent turns: 4. Signals seen: feedback_acknowledged, information_provided. Evidence: [tweet 1236515, feedback_acknowledged] "We'll pass your comments on in regards to the card machine."

Fill in CSV row `case_1236520`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 137 · `case_891542`

**First customer message:** @VirginTrains are you able to look into a delay repay query from 15/9/17, I have VT no. The case was reopened a week ago, I've heard nothing

**Conversation**

1. **CUSTOMER** `891542` 2017-10-13 17:50: @VirginTrains are you able to look into a delay repay query from 15/9/17, I have VT no. The case was reopened a week ago, I've heard nothing
2. **AGENT** `891540` 2017-10-13 17:56: @331705 Hi Annii, can you confirm your email address so we can find you in our system please? ^HP
3. **CUSTOMER** `891541` 2017-10-13 18:15: @VirginTrains shall I private message you?
4. **AGENT** `891543` 2017-10-13 18:24: @331705 Do feel free to drop us a message Annii ^HP
5. **CUSTOMER** `891544` 2017-10-13 18:54: @VirginTrains just done, thanks

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `customer_service_complaint`, margin 0.1255)  
> Auto resolution: `redirected_to_dm`, resolved: True, DM redirect: True, outcome `customer_confirmed`  
> Type: redirected_to_dm. Outcome: customer_confirmed. Brand agent turns: 2. Signals seen: redirected_to_dm, clarification_requested. Evidence: [tweet 891543, redirected_to_dm] "Do feel free to drop us a message Annii"

Fill in CSV row `case_891542`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 138 · `case_2887025`

**First customer message:** @VirginTrains if I travel on your service on a pre paid ticket, and the station doesn’t have a machine to print my ticket of, how can your smug train guard charge me again? Disgrace ! I’ve paid for a service which hasn’t been provided.

**Conversation**

1. **CUSTOMER** `2887025` 2017-11-28 12:50: @VirginTrains if I travel on your service on a pre paid ticket, and the station doesn’t have a machine to print my ticket of, how can your smug train guard charge me again? Disgrace ! I’ve paid for a service which hasn’t been provided.
2. **AGENT** `2887022` 2017-11-28 12:51: @800627 If you've selected to collect your tickets before travel you do need to do this. If you travel without your ticket staff would be following the correct procedure in charging for a new fare ^CB
3. **CUSTOMER** `2887024` 2017-11-28 13:02: @VirginTrains Considering the station had no ticket booth the guard should of wavered the fact I couldn’t print it off and used my reference number and treated it as an E ticket, anyway maybe she can use her wages to buy a Diet plan or a life coach to sort her stinking attitude out.
4. **AGENT** `2887026` 2017-11-28 13:03: @800627 If you're unhappy with your experience and wish to make a complaint regarding this please visit our website: https://t.co/t20UbyOCSn ^CB
5. **CUSTOMER** `2887023` 2017-11-28 13:03: @VirginTrains Terrible treatment of a customer, a journey costing 100 has now cost me 188, because your train guard couldn’t let me print my ticket of in London.

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `ticket_booking_query`, margin 0.0163)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: self_service. Outcome: customer_followup_unanswered. Brand agent turns: 2. Signals seen: self_service, information_provided. Evidence: [tweet 2887026, self_service] "If you're unhappy with your experience and wish to make a complaint regarding this please visit our website: https://t.co/t20UbyOCSn"

Fill in CSV row `case_2887025`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 139 · `case_1489370`

**First customer message:** @VirginTrains Good Morning. How long does it take to deal with complaints as I’ve e-mailed customer services Monday and had no response?

**Conversation**

1. **CUSTOMER** `1489370` 2017-11-03 08:21: @VirginTrains Good Morning. How long does it take to deal with complaints as I’ve e-mailed customer services Monday and had no response?
2. **AGENT** `1489369` 2017-11-03 08:32: @465928 Hi there, our Customer Resolutions team aim to respond within 14 days, 28 max, so you should hear back shortly ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `delay_repay_refund_claim`, margin 0.0711)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 1489369, information_provided] "Hi there, our Customer Resolutions team aim to respond within 14 days, 28 max, so you should hear back shortly"

Fill in CSV row `case_1489370`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 140 · `case_2224882`

**First customer message:** @Virgintrains 3. I also have fibromyalgia and part of this is extreme sensitivity to noise and light. This is why I travel in the quietcoach

**Conversation**

1. **CUSTOMER** `2224882` 2017-10-19 15:43: @Virgintrains 3. I also have fibromyalgia and part of this is extreme sensitivity to noise and light. This is why I travel in the quietcoach
2. **AGENT** `2224879` 2017-10-19 15:45: @649565 1/2 Hi Charlotte, really sorry to hear that it's not as quiet as expected onboard. We would advise speaking to a member of
3. **AGENT** `2224883` 2017-10-19 15:45: @649565 2/2 the onboard team, as they may be able to put a friendly announcement out for you ^HP
4. **CUSTOMER** `2224880` 2017-10-19 15:49: @VirginTrains It’s never as quiet as expected. It’s a farce of a service now. I can say with certainty after 15 journeys in 5 weeks. Need more signs.
5. **CUSTOMER** `2224881` 2017-10-19 15:49: @VirginTrains More training for staff. More announcements. Why do I have to get up and find an onboard member of staff? I’ve just told you I have a
6. **CUSTOMER** `2224878` 2017-10-19 15:50: @VirginTrains Chronic pain condition.If you can’t do it yourselves then take it away #virgintrains #quietcoach #farce #quietzone #disability #fibromyalgia
7. **AGENT** `2224877` 2017-10-19 15:53: @649565 We'll certainly pass your comments on, Charlotte ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `customer_service_complaint`, margin 0.0404)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: feedback_acknowledged. Outcome: agent_answered_unconfirmed. Brand agent turns: 3. Signals seen: feedback_acknowledged, information_provided. Evidence: [tweet 2224877, feedback_acknowledged] "We'll certainly pass your comments on, Charlotte"

Fill in CSV row `case_2224882`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 141 · `case_1686858`

**First customer message:** @VirginTrains 1 hour delay from Carlisle.... not cool

**Conversation**

1. **CUSTOMER** `1686858` 2017-10-17 16:21: @VirginTrains 1 hour delay from Carlisle.... not cool
2. **AGENT** `1686855` 2017-10-17 16:30: @490937 Apologies Luke, which service are you waiting for? ^HP
3. **CUSTOMER** `1686857` 2017-10-17 16:50: @VirginTrains https://t.co/KJIM9DCdSD
4. **AGENT** `1686860` 2017-10-17 16:53: @490937 Apologies Luke, this service is currently 66 minutes late due to a fault on a train ahead of this service ^HP
5. **AGENT** `1686859` 2017-10-17 16:54: @490937 You can make a claim for compensation for your delay here: https://t.co/TluNccH9Su ^HP
6. **CUSTOMER** `1686856` 2017-10-17 16:56: @VirginTrains https://t.co/H12zYHSssP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 2 `delay_in_progress_report`; runner-up `service_status_delay_enquiry`, margin 0.0262)  
> Auto resolution: `compensation`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: compensation. Outcome: customer_followup_unanswered. Brand agent turns: 3. Signals seen: compensation, self_service, information_provided, clarification_requested. Evidence: [tweet 1686859, compensation] "You can make a claim for compensation for your delay here: https://t.co/TluNccH9Su"

Fill in CSV row `case_1686858`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 142 · `case_681179`

**First customer message:** @VirginTrains can you reply to my dm please

**Conversation**

1. **CUSTOMER** `681179` 2017-11-23 08:59: @VirginTrains can you reply to my dm please
2. **AGENT** `681178` 2017-11-23 09:03: @282559 We'll respond to your DM as soon as possible, Paige. We are receiving a high volume of messages at the moment due to multiple disruption ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `chitchat_non_support`, margin 0.0306)  
> Auto resolution: `redirected_to_dm`, resolved: False, DM redirect: True, outcome `redirected_to_dm`  
> Type: redirected_to_dm. Outcome: redirected_to_dm. Brand agent turns: 1. Signals seen: redirected_to_dm. Evidence: [tweet 681178, redirected_to_dm] "We'll respond to your DM as soon as possible, Paige."

Fill in CSV row `case_681179`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 143 · `case_2489585`

**First customer message:** I feel like this is contradictory to the trade descriptions act @VirginTrains https://t.co/lO6GQrL1Dq

**Conversation**

1. **CUSTOMER** `2489585` 2017-11-26 19:13: I feel like this is contradictory to the trade descriptions act @VirginTrains https://t.co/lO6GQrL1Dq
2. **AGENT** `2489584` 2017-11-26 19:14: @710890 Anything we can help with? ^MW
3. **CUSTOMER** `2489583` 2017-11-26 19:16: @VirginTrains No, just found it funny. A little part of me hoped CB was on shift so I could wind them up a bit but I’ll take the matter up with them personally.
4. **AGENT** `2489582` 2017-11-26 19:16: @710890 Fair play ^MW

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `chitchat_non_support`, margin 0.0105)  
> Auto resolution: `other`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: other. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: other. Evidence: [tweet 2489582, other] "Fair play"

Fill in CSV row `case_2489585`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 144 · `case_687380`

**First customer message:** Shocking service from Kings Cross to Peterborough. @VirginTrains arrived half an hour late meaning I missed my connecting train to Lincoln. Staff unfriendly and completely unhelpful at Peterborough station!

**Conversation**

1. **CUSTOMER** `687380` 2017-11-23 14:00: Shocking service from Kings Cross to Peterborough. @VirginTrains arrived half an hour late meaning I missed my connecting train to Lincoln. Staff unfriendly and completely unhelpful at Peterborough station!
2. **AGENT** `687377` 2017-11-23 14:06: @284163 Hi Sam, sorry to hear this. It sounds like you were travelling with @120576 today so we'll pass this over to the team for you ^HP
3. **CUSTOMER** `687378` 2017-11-23 14:12: @VirginTrains @120576 Fine. X
4. **OTHER-AGENT** `687379` 2017-11-23 14:15: @VirginTrains @284163 Hi Sam, I'm really sorry to hear this. You can make a claim for this online: https://t.co/5xmhxM2gcK. ^JC
5. **CUSTOMER** `687381` 2017-11-23 15:04: @120576 @VirginTrains Thanks so much xxx

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 1 `service_running_enquiry`; runner-up `service_status_delay_enquiry`, margin 0.013)  
> Auto resolution: `redirected_to_other_operator`, resolved: True, DM redirect: False, outcome `customer_confirmed`  
> Type: redirected_to_other_operator. Outcome: customer_confirmed. Brand agent turns: 1. Signals seen: redirected_to_other_operator. Evidence: [tweet 687377, redirected_to_other_operator] "It sounds like you were travelling with @120576 today so we'll pass this over to the team for you"

Fill in CSV row `case_687380`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 145 · `case_900189`

**First customer message:** So now my @virgintrains train’s delayed and the onboard reservations aren’t working! Welcome to pre-Brexit UK, it’ll all be so much better in 2020.

**Conversation**

1. **CUSTOMER** `900189` 2017-11-17 19:17: So now my @virgintrains train’s delayed and the onboard reservations aren’t working! Welcome to pre-Brexit UK, it’ll all be so much better in 2020.
2. **AGENT** `900188` 2017-11-17 19:19: @333709 Which service are you on Rik? ^PA
3. **CUSTOMER** `900187` 2017-11-17 19:19: @VirginTrains 1903 Euston-BNS
4. **CUSTOMER** `900185` 2017-11-17 19:20: @333709 @VirginTrains any news in when it’s departing?? No info here
5. **CUSTOMER** `900186` 2017-11-17 19:22: @333709 @VirginTrains it’s ok it’s left now...
6. **AGENT** `900184` 2017-11-17 19:23: @333709 @333709 It should be departing soon. ^A

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 1 `service_running_enquiry`; runner-up `journey_disruption_complaint`, margin 0.0011)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: information_provided, clarification_requested. Evidence: [tweet 900184, information_provided] "It should be departing soon. ^A"

Fill in CSV row `case_900189`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 146 · `case_2462577`

**First customer message:** @VirginTrains WLV to BTM today was an absolute joke, plenty of space, failure to manage meant several couldn't get on.

**Conversation**

1. **CUSTOMER** `2462577` 2017-11-15 08:21: @VirginTrains WLV to BTM today was an absolute joke, plenty of space, failure to manage meant several couldn't get on.
2. **AGENT** `2462578` 2017-11-15 08:33: @704963 Hi Ellie, it doesn't sound like you were travelling on our service today - could you just confirm which stations these are? ^HP
3. **CUSTOMER** `2462576` 2017-11-15 08:36: @VirginTrains Wolverhampton station - virgin managed, the management of the station at fault here. Manage customers expectations and safety concurrently.
4. **AGENT** `2462575` 2017-11-15 08:45: @704963 We'll be sure to pass your comments on about this, Ellie ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `chitchat_non_support`, margin 0.028)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: feedback_acknowledged. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: feedback_acknowledged, clarification_requested. Evidence: [tweet 2462575, feedback_acknowledged] "We'll be sure to pass your comments on about this, Ellie"

Fill in CSV row `case_2462577`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 147 · `case_2475457`

**First customer message:** @VirginTrains tell me why this is sensible? 1400 MK to Euston is running late but I am on allowed to get on the 1347 service which is running to time #jobsworths

**Conversation**

1. **CUSTOMER** `2475457` 2017-11-15 13:42: @VirginTrains tell me why this is sensible? 1400 MK to Euston is running late but I am on allowed to get on the 1347 service which is running to time #jobsworths
2. **AGENT** `2475456` 2017-11-15 13:44: @707666 Are you travelling with an Advance ticket for the 14:00 service? ^CB
3. **CUSTOMER** `2475455` 2017-11-15 13:45: @VirginTrains Yes and as it's running late thought you lot could be a bit flexible. Ha!
4. **AGENT** `2475454` 2017-11-15 13:46: @707666 With that ticket it's only valid on the booked train. As the service is still running you wouldn't be able to board another ^CB
5. **CUSTOMER** `2475453` 2017-11-15 13:49: @VirginTrains Even if it's late? Be honest don't you think that's a little daft (and extremely frustrating for your customer?)
6. **AGENT** `2475451` 2017-11-15 13:50: @707666 I understand the frustration, but it's part of the terms and conditions of the Advance fare ^CB
7. **CUSTOMER** `2475452` 2017-11-15 13:55: @VirginTrains Well I suggest you escalate this and get the powers that be have a think about your Ts and Vs. It's beyond ridiculous!

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 9 `service_status_update_request`; runner-up `service_status_delay_enquiry`, margin 0.0791)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: information_provided. Outcome: customer_followup_unanswered. Brand agent turns: 3. Signals seen: information_provided, clarification_requested. Evidence: [tweet 2475454, information_provided] "With that ticket it's only valid on the booked train. As the service is still running you wouldn't be able to board another"; [tweet 2475451, information_provided] "I understand the frustration, but it's part of the terms and conditions of the Advance fare"

Fill in CSV row `case_2475457`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 148 · `case_1748990`

**First customer message:** @VirginTrains currently on hold to customer service for the past 26 minutes!! 😟Just to ask a question re delayed train yesterday. HELP!!!

**Conversation**

1. **CUSTOMER** `1748990` 2017-11-07 16:17: @VirginTrains currently on hold to customer service for the past 26 minutes!! 😟Just to ask a question re delayed train yesterday. HELP!!!
2. **AGENT** `1748991` 2017-11-07 16:19: @527174 Which number are you calling today? ^MW
3. **CUSTOMER** `1748989` 2017-11-07 16:22: @VirginTrains 31 minutes now! 😡
4. **AGENT** `1748988` 2017-11-07 16:25: @527174 Thanks, it looks like you're trying to contact @120576 so we'll pass this over to the team for you ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `service_status_delay_enquiry`, margin 0.0183)  
> Auto resolution: `redirected_to_other_operator`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: redirected_to_other_operator. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: redirected_to_other_operator, clarification_requested. Evidence: [tweet 1748988, redirected_to_other_operator] "Thanks, it looks like you're trying to contact @120576 so we'll pass this over to the team for you"

Fill in CSV row `case_1748990`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 149 · `case_56496`

**First customer message:** 1) Yo @VirginTrains just want you to know, I hung around waiting for your "cheaper" off peak ticket (£90/opposed to £145) &amp; I get to sit...

**Conversation**

1. **CUSTOMER** `56496` 2017-11-01 19:31: 1) Yo @VirginTrains just want you to know, I hung around waiting for your "cheaper" off peak ticket (£90/opposed to £145) &amp; I get to sit...
2. **AGENT** `56497` 2017-11-01 19:35: @128784 1/2 Our Best Fares are released 12 weeks in advance for weekends and 24 weeks for weekdays
3. **AGENT** `56495` 2017-11-01 19:35: @128784 2/2 Alternatively you can search via our Best Fare Finder - https://t.co/JDdH4kPDC9 ^MM
4. **CUSTOMER** `56500` 2017-11-01 19:39: @VirginTrains Sure I'll just plan all my LDN meetings 24 weeks in advance in future.... spare me. It's extortion plain &amp; simple.
5. **CUSTOMER** `56499` 2017-11-01 19:40: @VirginTrains Do tell me your "best fare" price for a peak ticket if I booked it 24 weeks in advance?
6. **AGENT** `56501` 2017-11-01 19:48: @128784 Where are you travelling to /from? ^MM
7. **CUSTOMER** `56502` 2017-11-01 19:49: @VirginTrains Stockport to London regularly.
8. **AGENT** `56503` 2017-11-01 19:53: @128784 We've got tickets from £23 (one way) via Our Best Fare Finder ^MM
9. **CUSTOMER** `56504` 2017-11-01 20:37: @VirginTrains What time of day?
10. **AGENT** `56505` 2017-11-01 20:39: @128784 These are throughout the day ^CB
11. **CUSTOMER** `56506` 2017-11-01 20:41: @VirginTrains Between 7am and 9:43??
12. **AGENT** `56507` 2017-11-01 20:50: @128784 Yes. Please see an example week attached. More details can be found via our Best Fare Finder - https://t.co/JDdH4kPDC9 ^MM https://t.co/gNkCUFR0iD
13. **CUSTOMER** `56508` 2017-11-01 20:53: @VirginTrains https://t.co/RF3DPi8oAI

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `seat_reservation_issue`, margin 0.0719)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: self_service. Outcome: customer_followup_unanswered. Brand agent turns: 6. Signals seen: self_service, information_provided, clarification_requested, other. Evidence: [tweet 56495, self_service] "Alternatively you can search via our Best Fare Finder - https://t.co/JDdH4kPDC9"; [tweet 56507, self_service] "More details can be found via our Best Fare Finder - https://t.co/JDdH4kPDC9 ^MM https://t.co/gNkCUFR0iD"

Fill in CSV row `case_56496`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 150 · `case_1603198`

**First customer message:** Again a @VirginTrains jobsworth gets another £55 out of my 21 year old daughter for a journey she had already paid for. Well done!!!

**Conversation**

1. **CUSTOMER** `1603198` 2017-11-05 07:26: Again a @VirginTrains jobsworth gets another £55 out of my 21 year old daughter for a journey she had already paid for. Well done!!!
2. **AGENT** `1603195` 2017-11-05 07:28: @492028 Not great to hear, Nigel, can you explain what's happened please? ^HP
3. **CUSTOMER** `1603196` 2017-11-05 07:31: @VirginTrains She bought a single ticket from Mcr to Lon, then paid 70p to upgrade for a return. Didn’t keep receipt as not told to.
4. **CUSTOMER** `1603197` 2017-11-05 07:33: @VirginTrains Told on train she needed receipt which may be true but just another example of your staff taking advantage of young people. Not 1st time
5. **AGENT** `1603200` 2017-11-05 07:36: @492028 1/2 Sorry to hear that, Nigel, I'm afraid you would need all parts of the ticket in order for the ticket to be valid but I can
6. **AGENT** `1603199` 2017-11-05 07:37: @492028 2/2 completely understand the frustration this has caused. Really sorry to hear of her experience ^HP
7. **CUSTOMER** `1603202` 2017-11-05 07:44: @VirginTrains It seems as though you make the ticketing as difficult as possible to take in extra funds. Which is ok I guess you are running a business!!
8. **CUSTOMER** `1603203` 2017-11-05 07:45: @VirginTrains What I find poor is when I make a mistake invariably your staff are reasonable. However when younger people involved you staff react
9. **CUSTOMER** `1603204` 2017-11-05 07:45: @VirginTrains differently. I follow Man City and live in London so use your service every 2 weeks or so. So have witnessed many such experiences.
10. **CUSTOMER** `1603201` 2017-11-05 07:45: @VirginTrains Next time get your staff to pick on a 54yo and treat them the same way as a 21 yo. Thanks for your help.
11. **AGENT** `1603205` 2017-11-05 08:02: @492028 Apologies again Nigel, we'll certainly pass your comments on about this ^HP
12. **CUSTOMER** `1603206` 2017-11-05 14:07: @VirginTrains She had all parts of the ticket just not the receipt.

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `chitchat_non_support`, margin 0.0083)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: feedback_acknowledged. Outcome: customer_followup_unanswered. Brand agent turns: 4. Signals seen: feedback_acknowledged, information_provided, clarification_requested. Evidence: [tweet 1603205, feedback_acknowledged] "Apologies again Nigel, we'll certainly pass your comments on about this"

Fill in CSV row `case_1603198`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 151 · `case_2203259`

**First customer message:** Seems lighting not included in price of @VirginTrains well not if you're in coach C on the 15.08 to York and they've mislaid coach E 🙄

**Conversation**

1. **CUSTOMER** `2203259` 2017-10-19 14:17: Seems lighting not included in price of @VirginTrains well not if you're in coach C on the 15.08 to York and they've mislaid coach E 🙄
2. **AGENT** `2203257` 2017-10-19 14:19: @644233 Hi Esmee, are you travelling on a @120576 service today? ^HP
3. **CUSTOMER** `2203258` 2017-10-19 14:21: @VirginTrains @120576 Yes.

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `praise_positive_feedback`, margin 0.107)  
> Auto resolution: `redirected_to_other_operator`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: redirected_to_other_operator. Outcome: customer_followup_unanswered. Brand agent turns: 1. Signals seen: redirected_to_other_operator, clarification_requested. Evidence: [tweet 2203257, redirected_to_other_operator] "Hi Esmee, are you travelling on a @120576 service today?"

Fill in CSV row `case_2203259`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 152 · `case_790427`

**First customer message:** @VirginTrains 😡😡😡😡 waiting on line for 90 mins to make complaint before being cut off

**Conversation**

1. **CUSTOMER** `790427` 2017-10-12 11:58: @VirginTrains 😡😡😡😡 waiting on line for 90 mins to make complaint before being cut off
2. **AGENT** `790426` 2017-10-12 12:01: @308485 Not great to hear, Stephen, which number did you contact please? ^HP
3. **CUSTOMER** `790425` 2017-10-12 12:07: @VirginTrains 0345 722 5111 Not happy at all! please email me with a direct number - __email__
4. **AGENT** `790423` 2017-10-12 12:12: @308485 Thanks for confirming, this is a @120576 number, so we'll pass your tweet over to the team for you ^HP
5. **CUSTOMER** `790424` 2017-10-12 12:46: @VirginTrains @120576 Still not heard back... I've been down to Leeds City Station and they have referred me to customer relations - Very poor!
6. **CUSTOMER** `790428` 2017-10-12 16:55: @VirginTrains I am very surprised and disappointed that Virgin Trains East Coast have not been in touch... Have they acknowledged the tweet?
7. **AGENT** `803709` 2017-10-12 16:58: @308485 We'll tag them again for you Stephen, @120576 can you please assist. ^PA
8. **CUSTOMER** `803710` 2017-10-12 17:45: @VirginTrains @120576 Thanks

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `service_status_delay_enquiry`, margin 0.0109)  
> Auto resolution: `redirected_to_other_operator`, resolved: True, DM redirect: False, outcome `customer_confirmed`  
> Type: redirected_to_other_operator. Outcome: customer_confirmed. Brand agent turns: 3. Signals seen: redirected_to_other_operator, information_provided, clarification_requested. Evidence: [tweet 790423, redirected_to_other_operator] "Thanks for confirming, this is a @120576 number, so we'll pass your tweet over to the team for you"

Fill in CSV row `case_790427`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 153 · `case_2317337`

**First customer message:** I’m not promoting the platform... just the team. @VirginTrains - one of the best social media outfits online. 👍🏼👏🏼👏🏼 https://t.co/UXZU7t8Esi

**Conversation**

1. **CUSTOMER** `2317337` 2017-10-19 23:00: I’m not promoting the platform... just the team. @VirginTrains - one of the best social media outfits online. 👍🏼👏🏼👏🏼 https://t.co/UXZU7t8Esi
2. **AGENT** `2317336` 2017-10-19 23:02: @283828 Thanks for the feedback ^RD

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `customer_service_complaint`, margin 0.0198)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: feedback_acknowledged. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: feedback_acknowledged. Evidence: [tweet 2317336, feedback_acknowledged] "Thanks for the feedback"

Fill in CSV row `case_2317337`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 154 · `case_156999`

**First customer message:** @VirginTrains just purchased the Internet on virgin trains. Should have known better. £5 and it doesn’t even work.

**Conversation**

1. **CUSTOMER** `156999` 2017-11-24 17:23: @VirginTrains just purchased the Internet on virgin trains. Should have known better. £5 and it doesn’t even work.
2. **AGENT** `156997` 2017-11-24 17:40: @152061 Can you get connected via https://t.co/1JEOSVPtGN ? ^MM
3. **CUSTOMER** `156998` 2017-11-24 17:41: @VirginTrains No
4. **AGENT** `157000` 2017-11-24 17:48: @152061 Please claim a refund via __email__ ^MM

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `onboard_wifi_issue` (cluster 11 `onboard_wifi_issue`; runner-up `journey_disruption_complaint`, margin 0.0399)  
> Auto resolution: `refund`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: refund. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: refund, compensation, self_service. Evidence: [tweet 157000, refund] "Please claim a refund via __email__"

Fill in CSV row `case_156999`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 155 · `case_1132500`

**First customer message:** Cheers @VirginTrains https://t.co/i8zMh3KWQ8

**Conversation**

1. **CUSTOMER** `1132500` 2017-10-24 12:17: Cheers @VirginTrains https://t.co/i8zMh3KWQ8
2. **AGENT** `1132499` 2017-10-24 12:19: @386869 Look at that! #Winning ^MW

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `unclear_or_media_only` (cluster -1 `unclear_or_media_only`; runner-up `unclear_or_media_only`, margin 0.0518)  
> Auto resolution: `other`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: other. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 1132499, other] "Look at that! #Winning"

Fill in CSV row `case_1132500`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 156 · `case_1524566`

**First customer message:** @VirginTrains on the 21:10 to london euston from new street, coach D... its smells like a sewage plant. The whooolleee coach

**Conversation**

1. **CUSTOMER** `1524566` 2017-11-03 21:12: @VirginTrains on the 21:10 to london euston from new street, coach D... its smells like a sewage plant. The whooolleee coach
2. **AGENT** `1524565` 2017-11-03 21:16: @473617 Oh no, have you been able to switch coaches at all? ^MW

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `praise_positive_feedback` (cluster 8 `praise_positive_feedback`; runner-up `seat_reservation_issue`, margin 0.0155)  
> Auto resolution: `clarification_requested`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: clarification_requested. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: clarification_requested. Evidence: [tweet 1524565, clarification_requested] "Oh no, have you been able to switch coaches at all?"

Fill in CSV row `case_1524566`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 157 · `case_2760189`

**First customer message:** @VirginTrains When children travel in rail replacement coaches, do they need car seats/must they have them/can they use them even if they haven't booked or paid for a seat (talking about a 1 and 3 year old here). Route is Reading - Winchester. Thx

**Conversation**

1. **CUSTOMER** `2760189` 2017-11-21 05:01: @VirginTrains When children travel in rail replacement coaches, do they need car seats/must they have them/can they use them even if they haven't booked or paid for a seat (talking about a 1 and 3 year old here). Route is Reading - Winchester. Thx
2. **AGENT** `2760187` 2017-11-21 05:04: @147083 Good question, we do not believe car seats are required however if you have a car seat you can use it if you wish ^JH
3. **CUSTOMER** `2760188` 2017-11-21 05:06: @VirginTrains Thanks. Can you get me a definite answer on the car seats as obviously being stranded with two small children would be less than ideal. Could you also confirm whether the 1 year old would be OK to travel on my lap (i.e. no seatbelt) - I assume the coaches are fitted with belts?
4. **AGENT** `2760190` 2017-11-21 05:15: @147083 There is no requirement for a car seat to be fitted. Children can travel with or without a car seat and children can travel on your lap if this is your preference ^JH
5. **CUSTOMER** `2760192` 2017-11-21 05:26: @VirginTrains Great, thank you
6. **AGENT** `2760196` 2017-11-21 05:27: @147083 You're welcome ^JH
7. **CUSTOMER** `2760191` 2017-11-21 05:27: @VirginTrains sorry - can you confirm the buses have seat belts? Obviously car seats don't work without them...
8. **AGENT** `2760193` 2017-11-21 05:30: @147083 All buses from 2001 have a three pin seat belt or lap belt which may or may not fit your child seat. We're sorry we have limited knowledge of the different seat belts available. ^JH
9. **CUSTOMER** `2760194` 2017-11-21 05:31: @VirginTrains thanks, my seat fits anything with a lap belt so that is v useful.
10. **AGENT** `2760195` 2017-11-21 05:31: @147083 Happy days ^JH

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `journey_disruption_complaint`, margin 0.0245)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 5. Signals seen: information_provided, other. Evidence: [tweet 2760187, information_provided] "Good question, we do not believe car seats are required however if you have a car seat you can use it if you wish"; [tweet 2760190, information_provided] "There is no requirement for a car seat to be fitted. Children can travel with or without a car seat and children can travel on your lap if this is your preference"; [tweet 2760193, information_provided] "All buses from 2001 have a three pin seat belt or lap belt which may or may not fit your child seat. We're sorry we have limited knowledge of the different seat belts available."

Fill in CSV row `case_2760189`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 158 · `case_2693657`

**First customer message:** @VirginTrains your service is a joke. Letting hundreds of people stand in the middle of the gangway annoying other passengers and making everyone feel uneasy. Pay all this money and its an awful service. Won’t be travelling again.

**Conversation**

1. **CUSTOMER** `2693657` 2017-11-19 17:15: @VirginTrains your service is a joke. Letting hundreds of people stand in the middle of the gangway annoying other passengers and making everyone feel uneasy. Pay all this money and its an awful service. Won’t be travelling again.
2. **AGENT** `2693655` 2017-11-19 17:19: @757096 Hi Daniel, which service are you on? ^HP
3. **CUSTOMER** `2693656` 2017-11-19 19:20: @VirginTrains Glasgow central - london euston 14.37 absolute joke
4. **AGENT** `2693658` 2017-11-19 19:29: @757096 Apologies Daniel, did you originally have a seat reservation for this service? ^HP
5. **CUSTOMER** `2693659` 2017-11-19 20:13: @VirginTrains We did yes. People kept asking to sit in our seats had people in the aisles next to ys for four hours. Not happy
6. **AGENT** `2693660` 2017-11-19 20:33: @757096 Sorry about your experience, Daniel, did you have to move at all? ^HP
7. **CUSTOMER** `2693661` 2017-11-19 20:43: @VirginTrains Well we felt like we had to as people had kids so i ended up standing as i felt bad. So gave up my own seat
8. **CUSTOMER** `2693662` 2017-11-20 10:19: @VirginTrains ??
9. **AGENT** `2693663` 2017-11-20 10:19: @757096 Hi Daniel. How can we help today? ^BT
10. **CUSTOMER** `2693664` 2017-11-20 15:19: @VirginTrains Read the feed please.
11. **AGENT** `2693665` 2017-11-20 15:25: @757096 Sorry to hear about your journey yesterday, Daniel. Did you manage to get a seat at all in the end? ^HP
12. **CUSTOMER** `2693666` 2017-11-20 15:30: @VirginTrains Yes we had reserved seats but gave them up when 100’s of people got on
13. **AGENT** `2693667` 2017-11-20 15:42: @757096 Apologies for the inconvenience caused, Daniel. You can claim compensation for not getting your seat by contacting our Customer Resolutions team via our online form here: https://t.co/t20UbyOCSn ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `journey_disruption_complaint`, margin 0.0373)  
> Auto resolution: `compensation`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: compensation. Outcome: redirected_to_channel. Brand agent turns: 6. Signals seen: compensation, self_service, clarification_requested, other. Evidence: [tweet 2693667, compensation] "You can claim compensation for not getting your seat by contacting our Customer Resolutions team via our online form here: https://t.co/t20UbyOCSn"

Fill in CSV row `case_2693657`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 159 · `case_755657`

**First customer message:** @VirginTrains Refunds dept, incompetence level: Master.

**Conversation**

1. **CUSTOMER** `755657` 2017-10-11 18:00: @VirginTrains Refunds dept, incompetence level: Master.
2. **AGENT** `755656` 2017-10-11 18:02: @300619 Anything we can help with? ^PA
3. **CUSTOMER** `755655` 2017-10-11 18:06: @VirginTrains Yes, employ c/centre staff who can either read or stick to the same excuse. 2hrs+ on phone and 4 e-mails to get refund is ridiculous.
4. **AGENT** `755654` 2017-10-11 18:07: @300619 I'll pass on your comments. ^PA

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `customer_service_complaint`, margin 0.1003)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: feedback_acknowledged. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: feedback_acknowledged, other. Evidence: [tweet 755654, feedback_acknowledged] "I'll pass on your comments."

Fill in CSV row `case_755657`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 160 · `case_1621715`

**First customer message:** SOS - Currently stuck on the @VirginTrains 14.40 Euston to Edinburgh with no M&amp;S picnic as we upgraded to 1st but no catering on board. 😭😱🤨

**Conversation**

1. **CUSTOMER** `1621715` 2017-11-05 16:53: SOS - Currently stuck on the @VirginTrains 14.40 Euston to Edinburgh with no M&amp;S picnic as we upgraded to 1st but no catering on board. 😭😱🤨
2. **AGENT** `1621712` 2017-11-05 16:59: @496910 Sorry for that Terry, are staff able to help at all. ^PA
3. **CUSTOMER** `1621713` 2017-11-05 17:26: @VirginTrains The manager is doing a super job! But just no catering on board. Will it come on board at Birmingham?
4. **CUSTOMER** `1621714` 2017-11-05 18:19: @VirginTrains We have tea and coffee, Yay! But turns out there’s no food🤨😱😭 Hungry 1st class passengers, Boo! #anyonegotasandwich?

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `first_class_catering_issue` (cluster 4 `first_class_service_issue`; runner-up `service_status_delay_enquiry`, margin 0.0005)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: information_provided. Outcome: customer_followup_unanswered. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 1621712, information_provided] "Sorry for that Terry, are staff able to help at all."

Fill in CSV row `case_1621715`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 161 · `case_580368`

**First customer message:** @VirginTrains hi, how do I go about getting a refund for my journey today? Booked tickets and reserved a seat for the 9.45 from Euston to Glasgow. Now having to stand for 5 and a half hours on 9.48 from Kings Cross to Edinburgh

**Conversation**

1. **CUSTOMER** `580368` 2017-12-03 09:50: @VirginTrains hi, how do I go about getting a refund for my journey today? Booked tickets and reserved a seat for the 9.45 from Euston to Glasgow. Now having to stand for 5 and a half hours on 9.48 from Kings Cross to Edinburgh
2. **AGENT** `580367` 2017-12-03 11:37: @256710 Apologies for your experience, Nicky. Unfortunately we're experiencing major disruption across our network due to emergency overhead line repairs. You can claim compensation for your journey by contacting our Customer Resolutions team here: https://t.co/qJCVixyLLR

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `ticket_booking_query`, margin 0.0589)  
> Auto resolution: `compensation`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: compensation. Outcome: redirected_to_channel. Brand agent turns: 1. Signals seen: compensation, self_service. Evidence: [tweet 580367, compensation] "You can claim compensation for your journey by contacting our Customer Resolutions team here: https://t.co/qJCVixyLLR"

Fill in CSV row `case_580368`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 162 · `case_2726549`

**First customer message:** @764613 @VirginTrains Wtf, deadass train bullshit

**Conversation**

1. **CUSTOMER** `2726549` 2017-11-20 13:12: @764613 @VirginTrains Wtf, deadass train bullshit
2. **AGENT** `2726548` 2017-11-20 13:18: @764614 @764613 Sorry to hear this. Can you please DM us your booking reference so I can look into this for you please? ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `service_status_delay_enquiry`, margin 0.0918)  
> Auto resolution: `redirected_to_dm`, resolved: False, DM redirect: True, outcome `redirected_to_dm`  
> Type: redirected_to_dm. Outcome: redirected_to_dm. Brand agent turns: 1. Signals seen: redirected_to_dm, escalated, clarification_requested. Evidence: [tweet 2726548, redirected_to_dm] "Can you please DM us your booking reference so I can look into this for you please?"

Fill in CSV row `case_2726549`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 163 · `case_635180`

**First customer message:** @VirginTrains @123241 so tired of being hung up on when all I need is advice that applies to a group of us. 😩

**Conversation**

1. **CUSTOMER** `635180` 2017-11-22 17:01: @VirginTrains @123241 so tired of being hung up on when all I need is advice that applies to a group of us. 😩
2. **AGENT** `635178` 2017-11-22 17:03: @270603 @123241 Hi Elenid, sorry to hear this, is there anything we can help with? ^HP
3. **CUSTOMER** `635179` 2017-11-22 17:18: @VirginTrains @123241 Trying to sort a group from all over the country for my sisters hendo. Wish I could get advice on tickets purchased and railcards available before some1 hangs up on me, realising I can’t buy immediately. Information given isn’t consistent, costing people extra ££ unnecessarily 😔

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `customer_service_complaint`, margin 0.0122)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: unresolved. Outcome: customer_followup_unanswered. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 635178, other] "Hi Elenid, sorry to hear this, is there anything we can help with?"

Fill in CSV row `case_635180`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 164 · `case_1396877`

**First customer message:** @VirginTrains how do I get a refund on a ticket? Only found out once I was on the train that my ticket wasn't valid on @120244 ??!!

**Conversation**

1. **CUSTOMER** `1396877` 2017-10-28 15:13: @VirginTrains how do I get a refund on a ticket? Only found out once I was on the train that my ticket wasn't valid on @120244 ??!!
2. **AGENT** `1396875` 2017-10-28 15:14: @444831 @120244 So have you travelled with the ticket? ^MW
3. **CUSTOMER** `1396876` 2017-10-28 15:16: @VirginTrains @120244 I had to buy a new one once I was on the train (to Chester from Manchester pic)
4. **AGENT** `1396878` 2017-10-28 15:18: @444831 @120244 I see, and is the ticket an Advance ticket? ^MW
5. **CUSTOMER** `1396879` 2017-10-28 15:18: @VirginTrains @120244 I've dm'd you all the details

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `ticket_booking_query`, margin 0.1107)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: unresolved. Outcome: customer_followup_unanswered. Brand agent turns: 2. Signals seen: clarification_requested, other. Evidence: [tweet 1396878, other] "I see, and is the ticket an Advance ticket?"

Fill in CSV row `case_1396877`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 165 · `case_2196800`

**First customer message:** Top service from Katy and Perry on @VirginTrains 👌🏼

**Conversation**

1. **CUSTOMER** `2196800` 2017-11-09 23:45: Top service from Katy and Perry on @VirginTrains 👌🏼
2. **AGENT** `2196801` 2017-11-09 23:50: @642686 That's great to hear Joel! We'll certainly pass on your kind words with a few 'fireworks' ;) ^SB https://t.co/0GPm2OCLNo
3. **AGENT** `2196799` 2017-11-09 23:53: @642686 Do you have details of the train you were on this evening? ^SB
4. **CUSTOMER** `2196798` 2017-11-10 00:00: @VirginTrains Euston to Birmingham New Street, left London at 2230pm.
5. **AGENT** `2196797` 2017-11-10 00:01: @642686 Thanks Joel. ^SB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `praise_positive_feedback`, margin 0.0241)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: self_service. Outcome: agent_answered_unconfirmed. Brand agent turns: 3. Signals seen: self_service, feedback_acknowledged, clarification_requested, other. Evidence: [tweet 2196801, self_service] "We'll certainly pass on your kind words with a few 'fireworks' ;) ^SB https://t.co/0GPm2OCLNo"

Fill in CSV row `case_2196800`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 166 · `case_1328090`

**First customer message:** @VirginTrains my daughter is traveling for the 1st time on her own to London Euston. Urgent update needed pls

**Conversation**

1. **CUSTOMER** `1328090` 2017-10-27 12:05: @VirginTrains my daughter is traveling for the 1st time on her own to London Euston. Urgent update needed pls
2. **AGENT** `1328089` 2017-10-27 12:14: @429854 Our services are unable to travel in to London Euston due to a person being hit by a train. ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 9 `service_status_update_request`; runner-up `service_status_delay_enquiry`, margin 0.0019)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 1328089, information_provided] "Our services are unable to travel in to London Euston due to a person being hit by a train."

Fill in CSV row `case_1328090`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 167 · `case_2514056`

**First customer message:** @VirginTrains good morning. How do we go about the compensation for a Train that was cancelled?

**Conversation**

1. **CUSTOMER** `2514056` 2017-10-30 07:57: @VirginTrains good morning. How do we go about the compensation for a Train that was cancelled?
2. **AGENT** `2514054` 2017-10-30 07:59: @465928 Which service was this? ^CB
3. **CUSTOMER** `2514055` 2017-10-30 08:00: @VirginTrains It was the 7:03pm from Euston to Wolverhampton on Saturday 28th October
4. **AGENT** `2514057` 2017-10-30 08:02: @465928 Please get in touch through our website: https://t.co/yRMQf6JYRw ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `journey_disruption_complaint`, margin 0.0502)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: self_service. Outcome: redirected_to_channel. Brand agent turns: 2. Signals seen: self_service, clarification_requested. Evidence: [tweet 2514057, self_service] "Please get in touch through our website: https://t.co/yRMQf6JYRw"

Fill in CSV row `case_2514056`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 168 · `case_2033366`

**First customer message:** @VirginTrains Jeez, been stuck in the car park at Bham Int for over 25 mins and not moved an inch. A little assistance maybe?! It's gridlock

**Conversation**

1. **CUSTOMER** `2033366` 2017-11-07 23:34: @VirginTrains Jeez, been stuck in the car park at Bham Int for over 25 mins and not moved an inch. A little assistance maybe?! It's gridlock
2. **AGENT** `2033363` 2017-11-07 23:36: @122561 Ah, there must've been a concert on... wait patiently is all I can suggest I'm afraid ^BH
3. **CUSTOMER** `2033365` 2017-11-07 23:38: @VirginTrains There was but this is ridiculous - nobody is moving. Can't even get out of the spaces!
4. **CUSTOMER** `2033364` 2017-11-07 23:39: @VirginTrains There has to be a better system!

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 2 `delay_in_progress_report`; runner-up `seat_reservation_issue`, margin 0.011)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: information_provided. Outcome: customer_followup_unanswered. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 2033363, information_provided] "Ah, there must've been a concert on... wait patiently is all I can suggest I'm afraid"

Fill in CSV row `case_2033366`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 169 · `case_1336797`

**First customer message:** @VirginTrains @431569 Today?! My train London to NE on Wednesday had people sitting in the aisles.. so many it was impossible to even get through to food car.

**Conversation**

1. **CUSTOMER** `1336797` 2017-10-27 14:59: @VirginTrains @431569 Today?! My train London to NE on Wednesday had people sitting in the aisles.. so many it was impossible to even get through to food car.
2. **CUSTOMER** `1336796` 2017-10-27 15:00: @VirginTrains @431569 Full refund should be given to anyone without a seat.
3. **CUSTOMER** `1336860` 2017-10-27 15:00: @431815 @VirginTrains @431569 Huh?

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `service_status_delay_enquiry`, margin 0.065)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `no_response`  
> Brand never replied in the public thread (outcome: no_response).

Fill in CSV row `case_1336797`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 170 · `case_1622396`

**First customer message:** @115793 @VirginTrains - please can you update those waiting on trains at Piccadilly. Should we stay or give up #MiltonKeynesCentral

**Conversation**

1. **CUSTOMER** `1622396` 2017-11-05 16:57: @115793 @VirginTrains - please can you update those waiting on trains at Piccadilly. Should we stay or give up #MiltonKeynesCentral
2. **AGENT** `1622395` 2017-11-05 17:06: @496914 @115793 We're awaiting an update as to when the lines will reopen. Live Updates can be found via https://t.co/Tpobzg1tWi ^MM

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `service_status_delay_enquiry`, margin 0.0088)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: self_service. Outcome: redirected_to_channel. Brand agent turns: 1. Signals seen: self_service. Evidence: [tweet 1622395, self_service] "Live Updates can be found via https://t.co/Tpobzg1tWi"

Fill in CSV row `case_1622396`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 171 · `case_1355619`

**First customer message:** @VirginTrains Taken 7 hours to get from London to Manchester sat on floor ,2 trains , one 30 min walk and 7000 stops. I want my money back

**Conversation**

1. **CUSTOMER** `1355619` 2017-10-27 19:09: @VirginTrains Taken 7 hours to get from London to Manchester sat on floor ,2 trains , one 30 min walk and 7000 stops. I want my money back
2. **AGENT** `1355618` 2017-10-27 19:13: @431004 We're so sorry for your experience, Pete. You can make a claim for compensation here: https://t.co/BwIfxpqVNk ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `journey_disruption_complaint`, margin 0.0615)  
> Auto resolution: `compensation`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: compensation. Outcome: redirected_to_channel. Brand agent turns: 1. Signals seen: compensation, self_service. Evidence: [tweet 1355618, compensation] "You can make a claim for compensation here: https://t.co/BwIfxpqVNk"

Fill in CSV row `case_1355619`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 172 · `case_2259295`

**First customer message:** love paying 70 quid to sit on the floor xx @VirginTrains 😘😘😘

**Conversation**

1. **CUSTOMER** `2259295` 2017-11-26 15:27: love paying 70 quid to sit on the floor xx @VirginTrains 😘😘😘
2. **AGENT** `2259293` 2017-11-26 15:27: @657906 Did you have a seat reserved for your trip this afternoon? ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `praise_positive_feedback`, margin 0.059)  
> Auto resolution: `clarification_requested`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: clarification_requested. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: clarification_requested. Evidence: [tweet 2259293, clarification_requested] "Did you have a seat reserved for your trip this afternoon?"

Fill in CSV row `case_2259295`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 173 · `case_1335585`

**First customer message:** @VirginTrains @431568 @431569 I suppose #RichardBranson won’t be along checking footage on CTV so he can give some Tory blog pictures of empty seats, then?

**Conversation**

1. **CUSTOMER** `1335585` 2017-10-27 15:08: @VirginTrains @431568 @431569 I suppose #RichardBranson won’t be along checking footage on CTV so he can give some Tory blog pictures of empty seats, then?

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `chitchat_non_support`, margin 0.0009)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `no_response`  
> Brand never replied in the public thread (outcome: no_response).

Fill in CSV row `case_1335585`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 174 · `case_1343070`

**First customer message:** @433234 @VirginTrains Get the 19 or 260 mate

**Conversation**

1. **CUSTOMER** `1343070` 2017-10-27 16:19: @433234 @VirginTrains Get the 19 or 260 mate
2. **CUSTOMER** `1343072` 2017-10-27 16:33: @433234 @VirginTrains I’ll shout you 31p pal

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 9 `service_status_update_request`; runner-up `service_status_delay_enquiry`, margin 0.0357)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `no_response`  
> Brand never replied in the public thread (outcome: no_response).

Fill in CSV row `case_1343070`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 175 · `case_511021`

**First customer message:** @237442 @VirginTrains They'll get you home no matter how, they have to.

**Conversation**

1. **CUSTOMER** `511021` 2017-12-01 22:26: @237442 @VirginTrains They'll get you home no matter how, they have to.
2. **CUSTOMER** `511019` 2017-12-01 22:27: @237442 @VirginTrains They should allow you to take an earlier one too if there is one.
3. **CUSTOMER** `511016` 2017-12-01 22:28: @237442 @VirginTrains Online is showing the 2210 delayed.
4. **CUSTOMER** `511013` 2017-12-01 22:31: @237442 @VirginTrains Cancelled between Wolves and New Street, starting from New Street https://t.co/BGpP4es3VV
5. **AGENT** `511012` 2017-12-01 22:33: @234899 @237442 That should run yes ^MW

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `customer_service_complaint`, margin 0.0176)  
> Auto resolution: `other`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: other. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 511012, other] "That should run yes"

Fill in CSV row `case_511021`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 176 · `case_1336862`

**First customer message:** @431812 @431815 @VirginTrains @431569 Exactly huh?

**Conversation**

1. **CUSTOMER** `1336862` 2017-10-27 17:25: @431812 @431815 @VirginTrains @431569 Exactly huh?

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `unclear_or_media_only` (cluster -1 `unclear_or_media_only`; runner-up `unclear_or_media_only`, margin 0.0518)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `no_response`  
> Brand never replied in the public thread (outcome: no_response).

Fill in CSV row `case_1336862`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 177 · `case_2770474`

**First customer message:** @VirginTrains was on your 12.49 from Carlisle on Friday. Please pass on to Lorraine and Angela that they were great. On your 12.30 from Euston now and only brunch! Knew I should have had a BK! 😂 #bringoutthesandwiches

**Conversation**

1. **CUSTOMER** `2770474` 2017-11-21 12:51: @VirginTrains was on your 12.49 from Carlisle on Friday. Please pass on to Lorraine and Angela that they were great. On your 12.30 from Euston now and only brunch! Knew I should have had a BK! 😂 #bringoutthesandwiches
2. **AGENT** `2770473` 2017-11-21 12:53: @622193 Thank you for letting us know, I'll get this passed on for you. ^PA
3. **CUSTOMER** `2770472` 2017-11-21 12:55: @VirginTrains Great! The fella was@great too but I can’t remember his name. 😂
4. **AGENT** `2770470` 2017-11-21 12:56: @622193 No problem, I'll try to work it out. ^PA
5. **CUSTOMER** `2770471` 2017-11-21 12:57: @VirginTrains 👍🏻

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `praise_positive_feedback` (cluster 8 `praise_positive_feedback`; runner-up `chitchat_non_support`, margin 0.0901)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: feedback_acknowledged. Outcome: customer_followup_unanswered. Brand agent turns: 2. Signals seen: feedback_acknowledged, information_provided. Evidence: [tweet 2770473, feedback_acknowledged] "Thank you for letting us know, I'll get this passed on for you."

Fill in CSV row `case_2770474`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 178 · `case_2556462`

**First customer message:** @VirginTrains I have just got off the 13:30 London Euston to Glasgow. I have left my ticket wallet on there. Was seat H42

**Conversation**

1. **CUSTOMER** `2556462` 2017-11-16 15:46: @VirginTrains I have just got off the 13:30 London Euston to Glasgow. I have left my ticket wallet on there. Was seat H42
2. **AGENT** `2556461` 2017-11-16 15:48: @429046 Hi Andrew, you will need to call our lost property office on, 03331 031 031 option 1, 3. Hope you're reunited soon 😄 ^BT
3. **CUSTOMER** `2556460` 2017-11-16 15:58: @VirginTrains Thank you. Hopefully someone will hand it in. Was hoping you had a way to get your guys on board to grab it
4. **AGENT** `2556459` 2017-11-16 16:03: @429046 Sorry we don't unfortunately. ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `ticket_booking_query`, margin 0.0069)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: information_provided, other. Evidence: [tweet 2556461, information_provided] "Hi Andrew, you will need to call our lost property office on, 03331 031 031 option 1, 3. Hope you're reunited soon 😄"

Fill in CSV row `case_2556462`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 179 · `case_1336818`

**First customer message:** @431794 @431797 @VirginTrains @431569 The only way to stop overcrowding and make sure everybody gets a seat is to only be able to board if your ticket was purchased in advance.

**Conversation**

1. **CUSTOMER** `1336818` 2017-10-28 18:36: @431794 @431797 @VirginTrains @431569 The only way to stop overcrowding and make sure everybody gets a seat is to only be able to board if your ticket was purchased in advance.
2. **CUSTOMER** `1336817` 2017-10-28 18:37: @431794 @431797 @VirginTrains @431569 Private companies have an obligation to fulfill their contract, if financed by the government and we hit a recession, transport suffers.
3. **CUSTOMER** `1336822` 2017-10-29 10:46: @431798 @431794 @431797 @VirginTrains @431569 More people than ever before are using trains. We also have the safest network in the world.
4. **CUSTOMER** `1336827` 2017-10-29 11:45: @431794 @431798 @431797 @VirginTrains @431569 Leaves mulch by track sides and give the impression to signalers a train has came to a stop as the wet leaves become conductive.
5. **CUSTOMER** `1336826` 2017-10-29 11:47: @431794 @431798 @431797 @VirginTrains @431569 Network Rail would love to cut down tress near rail lines, but the public say no as they also block out the noise of trains
6. **CUSTOMER** `1336829` 2017-10-29 13:14: @431797 @431794 @431798 @VirginTrains @431569 I can travel on an advance purchase ticket cheaper now than in 1992. Irvine-Lancaster then, £27, now as cheap as £13
7. **CUSTOMER** `1336848` 2017-10-29 13:21: @431797 @431794 @431798 @VirginTrains @431569 Exactly, no competition. I can even get to Runcorn via Crewe for £13 sometimes, mainly £18.50
8. **CUSTOMER** `1336835` 2017-10-29 13:30: @431794 @431797 @431798 @VirginTrains @431569 Excuse me, but I've been travelling by train on a regular basis since '92, I think I'd know! I lived and worked in Lancashire. I still visit
9. **CUSTOMER** `1336834` 2017-10-29 13:35: @431794 @431797 @431798 @VirginTrains @431569 My partners parents live in Cheshire, I visit at least 6 times a year. I picked this date at random, it's about 240 miles one way https://t.co/ONkgFTPIzt
10. **CUSTOMER** `1336842` 2017-10-29 13:36: @431794 @431797 @431798 @VirginTrains @431569 No, but I do have family in the railways now and have done since Victorian times. I don't just travel to Lancashire and Cheshire either
11. **CUSTOMER** `1336843` 2017-10-29 13:37: @431794 @431797 @431798 @VirginTrains @431569 I haven't exactly not noticed the changes in that time, even locally. I have been invited to meetings by ScotRail management
12. **CUSTOMER** `1336837` 2017-10-29 13:39: @431794 @431797 @431798 @VirginTrains @431569 I can't prove how much the tickets were back then, I can remember personally, but I can for travelling in the near future.
13. **CUSTOMER** `1336845` 2017-10-29 13:40: @431794 @431797 @431798 @VirginTrains @431569 I'm proving a point, you don't seem to be able to handle that. Nationalisation means no competition and increasing fares as they please
14. **CUSTOMER** `1336839` 2017-10-29 13:41: @431794 @431797 @431798 @VirginTrains @431569 Here will do, I prefer it
15. **CUSTOMER** `1336849` 2017-10-29 13:55: @431797 @431794 @431798 @VirginTrains @431569 As I said, more travelling now than before https://t.co/ezpyskLiP7 NI Rail is nationalised, fares are at times going up at 3 times inflation
16. **CUSTOMER** `1336852` 2017-10-29 22:36: @431798 @431797 @431794 @VirginTrains @431569 From here to Cheshire / Liverpool have ScotRail / Virgin / Transpennine / Northern or Virgin / Arriva Wales/ London Midland sharing
17. **CUSTOMER** `1336851` 2017-10-29 22:40: @431798 @431797 @431794 @VirginTrains @431569 From here to Carlisle I have a choice of ScotRail, Virgin or TransPennine. Warrington to Runcorn can be Northern / Arriva / LM or Virgin.

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `seat_reservation_issue`, margin 0.0239)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `no_response`  
> Brand never replied in the public thread (outcome: no_response).

Fill in CSV row `case_1336818`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 180 · `case_815769`

**First customer message:** @VirginTrains @314246 Why have your Saturday RTN fares increased from £45 to £83.90 MAN-EUS? #discusting

**Conversation**

1. **CUSTOMER** `815769` 2017-10-08 20:20: @VirginTrains @314246 Why have your Saturday RTN fares increased from £45 to £83.90 MAN-EUS? #discusting
2. **AGENT** `815768` 2017-10-08 20:21: @314245 What fares are you looking at? ^CB

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `customer_service_complaint`, margin 0.0232)  
> Auto resolution: `clarification_requested`, resolved: False, DM redirect: False, outcome `agent_awaiting_customer`  
> Type: clarification_requested. Outcome: agent_awaiting_customer. Brand agent turns: 1. Signals seen: clarification_requested. Evidence: [tweet 815768, clarification_requested] "What fares are you looking at?"

Fill in CSV row `case_815769`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 181 · `case_756988`

**First customer message:** Thankyou @VirginTrains for not informing me of a change of platform, making me miss my advance booked trains from chorley to Leicester

**Conversation**

1. **CUSTOMER** `756988` 2017-10-11 18:35: Thankyou @VirginTrains for not informing me of a change of platform, making me miss my advance booked trains from chorley to Leicester
2. **CUSTOMER** `756984` 2017-10-11 18:36: @VirginTrains @VirginTrains your staff were so clearly keen to help, I pretty much got nothing but a grunt from them
3. **AGENT** `756980` 2017-10-11 18:41: @300908 Sounds like a @120576 service. ^PA
4. **CUSTOMER** `756983` 2017-10-11 18:45: @VirginTrains @120576 To be honest it sounds like I've just blown 40 odd quid to be able to get out of Preston
5. **CUSTOMER** `756987` 2017-10-11 18:47: @VirginTrains Also, it's definitely a west coast service
6. **AGENT** `757521` 2017-10-11 18:49: @300908 Where were you travelling? We don't run from Chorley to Leicester. ^PA
7. **CUSTOMER** `757520` 2017-10-11 18:50: @VirginTrains I booked advance tickets to travel from chorley to Leicester via Preston-crewe-nuneaton.
8. **OTHER-AGENT** `756982` 2017-10-11 18:51: @VirginTrains @300908 We don't call at either of those stations, not one of ours, sorry! ^JW
9. **AGENT** `757519` 2017-10-11 18:54: @300908 Ah I see, sorry for the confusion, it's been a long shift 😞 . ^PA
10. **CUSTOMER** `757518` 2017-10-11 18:55: @VirginTrains All I really expect is a little communication when a replacement train is waiting at a different platform
11. **AGENT** `757516` 2017-10-11 18:58: @300908 I appreciate that, sorry there wasn't this time. ^PA
12. **CUSTOMER** `757517` 2017-10-11 18:59: @VirginTrains I'm not made of money unfortunately, so forking out money for more tickets because of delays is kind of heartbreaking
13. **CUSTOMER** `756986` 2017-10-11 22:51: @300910 @120576 @VirginTrains K

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `service_status_delay_enquiry`, margin 0.0274)  
> Auto resolution: `redirected_to_other_operator`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: redirected_to_other_operator. Outcome: customer_followup_unanswered. Brand agent turns: 4. Signals seen: redirected_to_other_operator, information_provided, clarification_requested. Evidence: [tweet 756980, redirected_to_other_operator] "Sounds like a @120576 service."

Fill in CSV row `case_756988`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 182 · `case_686211`

**First customer message:** @VirginTrains I booked a ticket via Virgin, but was delayed over an hour because of Northern, do I reclaim of you or them?

**Conversation**

1. **CUSTOMER** `686211` 2017-11-23 13:38: @VirginTrains I booked a ticket via Virgin, but was delayed over an hour because of Northern, do I reclaim of you or them?
2. **AGENT** `686209` 2017-11-23 13:41: @283823 You'd need to claim with Northern for this ^CB
3. **CUSTOMER** `686210` 2017-11-23 13:42: @VirginTrains Ok, thought so

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `delay_repay_refund_claim`, margin 0.0561)  
> Auto resolution: `compensation`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: compensation. Outcome: customer_followup_unanswered. Brand agent turns: 1. Signals seen: compensation. Evidence: [tweet 686209, compensation] "You'd need to claim with Northern for this"

Fill in CSV row `case_686211`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 183 · `case_29131`

**First customer message:** I'm getting hangry...no food or drink offered yet and a delay with @VirginTrains over two hours into my journey to London!!

**Conversation**

1. **CUSTOMER** `29131` 2017-11-01 11:03: I'm getting hangry...no food or drink offered yet and a delay with @VirginTrains over two hours into my journey to London!!
2. **AGENT** `29129` 2017-11-01 11:05: @122408 Which service is it? ^PA
3. **CUSTOMER** `29130` 2017-11-01 11:11: @VirginTrains The 8.38 from Durham which didn't leave Durham till after 12 minutes past 9 as it was delayed!

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 2 `delay_in_progress_report`; runner-up `praise_positive_feedback`, margin 0.0151)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: unresolved. Outcome: customer_followup_unanswered. Brand agent turns: 1. Signals seen: clarification_requested. Evidence: [tweet 29129, clarification_requested] "Which service is it?"

Fill in CSV row `case_29131`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 184 · `case_2163906`

**First customer message:** @VirginTrains are the rail strikes today affecting trains out of London Euston at all? Thanks :)

**Conversation**

1. **CUSTOMER** `2163906` 2017-11-09 09:29: @VirginTrains are the rail strikes today affecting trains out of London Euston at all? Thanks :)
2. **AGENT** `2163904` 2017-11-09 09:30: @528171 Nope, we're unaffected, Ellie ^MW
3. **CUSTOMER** `2163905` 2017-11-09 09:37: @VirginTrains Thank you!

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `service_status_delay_enquiry` (cluster 1 `service_running_enquiry`; runner-up `journey_disruption_complaint`, margin 0.0036)  
> Auto resolution: `other`, resolved: True, DM redirect: False, outcome `customer_confirmed`  
> Type: other. Outcome: customer_confirmed. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 2163904, other] "Nope, we're unaffected, Ellie"

Fill in CSV row `case_2163906`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 185 · `case_2017632`

**First customer message:** @VirginTrains where u from BH

**Conversation**

1. **CUSTOMER** `2017632` 2017-11-07 22:47: @VirginTrains where u from BH
2. **AGENT** `2017630` 2017-11-07 22:49: @583236 Why? ^BH
3. **CUSTOMER** `2017631` 2017-11-07 22:50: @VirginTrains Just asking I'm from Plymouth
4. **CUSTOMER** `2017633` 2017-11-07 22:51: @VirginTrains On internal system find out where's 390155 currently at
5. **CUSTOMER** `2017634` 2017-11-07 22:57: @VirginTrains Bye now

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `praise_positive_feedback`, margin 0.0385)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: unresolved. Outcome: customer_followup_unanswered. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 2017630, other] "Why?"

Fill in CSV row `case_2017632`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 186 · `case_2453675`

**First customer message:** @VirginTrains absolutely disgusted your not replying to me why I’m sat on the floor of one of your overcrowded trains after iv spent £88

**Conversation**

1. **CUSTOMER** `2453675` 2017-10-29 18:47: @VirginTrains absolutely disgusted your not replying to me why I’m sat on the floor of one of your overcrowded trains after iv spent £88
2. **AGENT** `2453673` 2017-10-29 18:51: @702792 We did reply to your earlier tweets, Matty ^MW
3. **CUSTOMER** `2453674` 2017-10-29 18:54: @VirginTrains So can you tell me how health and safety is ok with 4 people sat by the door?????
4. **AGENT** `2453689` 2017-10-29 18:56: @702792 It's not what we like to hear or see but we can't stop passengers sitting there ^MW
5. **CUSTOMER** `2453670` 2017-10-29 18:56: @VirginTrains The 3 carriages of first class is empty but we pay and still sit on the floor must be a fire regulation am I right????
6. **CUSTOMER** `2453690` 2017-10-29 18:58: @VirginTrains But theirs no spare seats where would virgin like us to sit???
7. **AGENT** `2453668` 2017-10-29 18:59: @702792 No Matty ^MW
8. **CUSTOMER** `2453669` 2017-10-29 18:59: @VirginTrains No what??? Are you saying it’s not against fire regulations???
9. **AGENT** `2453691` 2017-10-29 19:01: @702792 The floor or standing would be the only option ^MW
10. **AGENT** `2453671` 2017-10-29 19:01: @702792 Not for passengers to sit on the floor, no ^MW
11. **CUSTOMER** `2453692` 2017-10-29 19:03: @VirginTrains So their are 4 people asleep by the doors of the train because their are no seats and your saying that’s not a breach of fire regulations???
12. **CUSTOMER** `2453672` 2017-10-29 19:05: @VirginTrains By the door asleep???
13. **AGENT** `2453693` 2017-10-29 19:05: @702792 I can't see how it would be but I'm not on the train, might be worth asking the onboard team ^MW
14. **CUSTOMER** `2453694` 2017-10-29 19:06: @VirginTrains But virgin I mean u are happy with us all sat on the floor. Do you think that is ok???
15. **AGENT** `2453695` 2017-10-29 19:08: @702792 We're not happy with it, no ^MW
16. **CUSTOMER** `2453697` 2017-10-29 19:09: @VirginTrains So what would you like to do about it?
17. **AGENT** `2453698` 2017-10-29 19:10: @702792 I'm afraid we can't do anything about this, Matty. If there's no seats free we can provide them for you ^MW
18. **CUSTOMER** `2453696` 2017-10-29 19:10: @VirginTrains How about you close 1 of ur 3 empty first class carriages and everyone can have a seat. EASY REALLY
19. **CUSTOMER** `2453699` 2017-10-29 19:11: @VirginTrains Y over book then
20. **AGENT** `2453704` 2017-10-29 19:12: @702792 We can't do that. Only the Train Manager has the authority to open First Class ^MW
21. **AGENT** `2453700` 2017-10-29 19:13: @702792 We don't. Overcrowding is caused by open tickets being used as they have high flexibility for travel ^MW
22. **CUSTOMER** `2453705` 2017-10-29 19:15: @VirginTrains And he can’t be contacted
23. **CUSTOMER** `2453702` 2017-10-29 19:16: @VirginTrains And can I ask when you book first class y is their 3 empty carriages. Surely u can see how many people have booked first class
24. **AGENT** `2453706` 2017-10-29 19:16: @702792 We can't authorise this or ask them to do this, no ^MW
25. **AGENT** `2453703` 2017-10-29 19:17: @702792 We can with Advance tickets, but again open tickets can be used at any time so we won't know how many will be used per service
26. **CUSTOMER** `2453701` 2017-10-29 19:19: @VirginTrains Absolute joke first class is never full just another money making scheme. Absolute joke and I can’t believe ur reply’s to passengers on flor

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `chitchat_non_support`, margin 0.1264)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: information_provided. Outcome: customer_followup_unanswered. Brand agent turns: 12. Signals seen: information_provided, other. Evidence: [tweet 2453673, information_provided] "We did reply to your earlier tweets, Matty"; [tweet 2453689, information_provided] "It's not what we like to hear or see but we can't stop passengers sitting there"; [tweet 2453691, information_provided] "The floor or standing would be the only option"; [tweet 2453671, information_provided] "Not for passengers to sit on the floor, no"

Fill in CSV row `case_2453675`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 187 · `case_1336783`

**First customer message:** @VirginTrains @431569 terrible service on Virgin NW Friday.No seats,whole families sitting in corridors.inhuman.utter misery.

**Conversation**

1. **CUSTOMER** `1336783` 2017-10-28 09:14: @VirginTrains @431569 terrible service on Virgin NW Friday.No seats,whole families sitting in corridors.inhuman.utter misery.

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `seat_reservation_issue`, margin 0.0187)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `no_response`  
> Brand never replied in the public thread (outcome: no_response).

Fill in CSV row `case_1336783`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 188 · `case_1441676`

**First customer message:** @VirginTrains https://t.co/HoSMtH1qmz

**Conversation**

1. **CUSTOMER** `1441676` 2017-11-02 12:24: @VirginTrains https://t.co/HoSMtH1qmz
2. **AGENT** `1441675` 2017-11-02 12:28: @454362 I see sorry for that, you can still upgrade if you wish, the staff onboard can assist with the prices. ^PA

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `unclear_or_media_only` (cluster -1 `unclear_or_media_only`; runner-up `unclear_or_media_only`, margin 0.0518)  
> Auto resolution: `information_provided`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: information_provided. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: information_provided. Evidence: [tweet 1441675, information_provided] "I see sorry for that, you can still upgrade if you wish, the staff onboard can assist with the prices."

Fill in CSV row `case_1441676`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 189 · `case_531889`

**First customer message:** Time is running out @VirginTrains. You acknowledged receipt of my letter of complaint on 06/11. Your 28 days expires on Monday! #Disappointing #PoorService

**Conversation**

1. **CUSTOMER** `531889` 2017-12-02 10:41: Time is running out @VirginTrains. You acknowledged receipt of my letter of complaint on 06/11. Your 28 days expires on Monday! #Disappointing #PoorService
2. **AGENT** `531888` 2017-12-02 10:46: @243223 Hi Michael, sorry to hear you've not received a response. Please contact our Customer Resolutions team via Live Chat: https://t.co/NHoA1QGqJm and they should be able to assist further ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `delay_repay_refund_claim`, margin 0.0242)  
> Auto resolution: `self_service`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: self_service. Outcome: redirected_to_channel. Brand agent turns: 1. Signals seen: self_service. Evidence: [tweet 531888, self_service] "Please contact our Customer Resolutions team via Live Chat: https://t.co/NHoA1QGqJm and they should be able to assist further"

Fill in CSV row `case_531889`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 190 · `case_1336790`

**First customer message:** @VirginTrains @431569 Correction; Sorry not sorry. If you were truly sorry things would change but they don't so it's just PR.

**Conversation**

1. **CUSTOMER** `1336790` 2017-10-27 21:03: @VirginTrains @431569 Correction; Sorry not sorry. If you were truly sorry things would change but they don't so it's just PR.

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `customer_service_complaint` (cluster 7 `customer_service_complaint`; runner-up `chitchat_non_support`, margin 0.0305)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `no_response`  
> Brand never replied in the public thread (outcome: no_response).

Fill in CSV row `case_1336790`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 191 · `case_1336792`

**First customer message:** @VirginTrains @431569 So you know its experiencing high volume of passengers so put on extr carriages or stop selling tickets ?

**Conversation**

1. **CUSTOMER** `1336792` 2017-10-27 23:59: @VirginTrains @431569 So you know its experiencing high volume of passengers so put on extr carriages or stop selling tickets ?

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `seat_reservation_issue`, margin 0.0331)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `no_response`  
> Brand never replied in the public thread (outcome: no_response).

Fill in CSV row `case_1336792`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 192 · `case_1620119`

**First customer message:** @VirginTrains got an advance single for tonight from ldn to Manchester, can I travel on any service when lines reopen?

**Conversation**

1. **CUSTOMER** `1620119` 2017-11-05 16:20: @VirginTrains got an advance single for tonight from ldn to Manchester, can I travel on any service when lines reopen?
2. **AGENT** `1620118` 2017-11-05 16:22: @495122 Yes you can. ^PA

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `service_status_delay_enquiry`, margin 0.0058)  
> Auto resolution: `other`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: other. Outcome: agent_answered_unconfirmed. Brand agent turns: 1. Signals seen: other. Evidence: [tweet 1620118, other] "Yes you can."

Fill in CSV row `case_1620119`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 193 · `case_874153`

**First customer message:** When @VirginTrains terminate your train early, close the buffet cart on the new train your only solace is a cute dog. https://t.co/tHQpjG3w6B

**Conversation**

1. **CUSTOMER** `874153` 2017-10-20 20:46: When @VirginTrains terminate your train early, close the buffet cart on the new train your only solace is a cute dog. https://t.co/tHQpjG3w6B
2. **AGENT** `874152` 2017-10-20 20:49: @121936 We're sorry for the inconvenience caused :( That is a very cute dog though 🐶 Which service are you on? ^HP
3. **CUSTOMER** `874146` 2017-10-20 20:54: @VirginTrains The 19:30 train to Glasgow from Euston. It’s a long time to go with no food or drink 😿
4. **AGENT** `874144` 2017-10-20 20:57: @121936 Apologies Alex, have you been able to speak to staff about this at all? ^HP
5. **CUSTOMER** `874145` 2017-10-20 21:00: @VirginTrains Staff are stressed and not sure what they’d say really. They just said they couldn’t operate a buffet car on this service. 2 hrs til Glasgow
6. **AGENT** `874147` 2017-10-20 21:05: @121936 Unfortunately we had to undertake a change of trains at CRE this evening which has led to this inconvenience ^RD
7. **CUSTOMER** `874148` 2017-10-20 21:07: @VirginTrains Understandable but I think it’s not much comfort to those of us who’ve had to endure overcrowding, chaos changing trains and now this.
8. **AGENT** `874149` 2017-10-20 21:10: @121936 If you speak with the Train Manager they may be able to help further ^RD
9. **CUSTOMER** `874150` 2017-10-20 21:14: @VirginTrains They’ve reopened! Hallelujah!
10. **AGENT** `874151` 2017-10-20 21:16: @121936 That's great to hear ^RD

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `journey_disruption_complaint` (cluster 5 `journey_disruption_complaint`; runner-up `service_status_delay_enquiry`, margin 0.0537)  
> Auto resolution: `information_provided`, resolved: True, DM redirect: False, outcome `agent_closed`  
> Type: information_provided. Outcome: agent_closed. Brand agent turns: 5. Signals seen: information_provided, clarification_requested, other. Evidence: [tweet 874147, information_provided] "Unfortunately we had to undertake a change of trains at CRE this evening which has led to this inconvenience"; [tweet 874149, information_provided] "If you speak with the Train Manager they may be able to help further"

Fill in CSV row `case_874153`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 194 · `case_64621`

**First customer message:** @VirginTrains can I get a refund if the strike goes ahead on the 22nd December? @130823 I have a return for the 24th

**Conversation**

1. **CUSTOMER** `64621` 2017-11-30 08:11: @VirginTrains can I get a refund if the strike goes ahead on the 22nd December? @130823 I have a return for the 24th
2. **AGENT** `64619` 2017-11-30 08:17: @130822 @130823 Yes, if you decide not to travel on the effected dates due to the strike a refund can be issued. ^BT
3. **CUSTOMER** `64610` 2017-11-30 08:18: @VirginTrains @130823 Will this refund cover the return journey
4. **AGENT** `64608` 2017-11-30 08:26: @130822 @130823 We would only issue a refund if customers decide to abandon their journey on the effected dates. ^BT
5. **CUSTOMER** `64609` 2017-11-30 08:35: @VirginTrains @130823 I have booked a return journey to London 22nd there and 24th return can you clarify if these dates are effected thanks
6. **AGENT** `64611` 2017-11-30 08:45: @130822 @130823 The 22nd December has been confirmed as a date effected by the strike. However we will run a majority of services on this date, but it will be a reduced service. ^BT
7. **CUSTOMER** `64612` 2017-11-30 09:36: @VirginTrains @130823 I'm a little confused can I claim a refund for the full amount of the return journey please
8. **AGENT** `64614` 2017-11-30 10:17: @130822 @130823 Only if you abandon travel Linze. ^BT
9. **CUSTOMER** `64615` 2017-11-30 10:18: @VirginTrains @130823 I'm sorry could you explain what you mean by that... do you mean cancel the whole train journey?
10. **AGENT** `64616` 2017-11-30 10:29: @130822 @130823 If this is a open return journey you will only receive a refund if the whole journey is abandoned. If these are Advance single tickets you can get a refund on the return advance single ticket if you do not travel. ^BT
11. **CUSTOMER** `64617` 2017-11-30 10:30: @VirginTrains @130823 Could you tell me where and how I apply for the refunds please
12. **AGENT** `64618` 2017-11-30 10:36: @130822 @130823 If you aren't travelling you can apply for a refund from point of purchase. For example if this was brought online, please log into your account and apply for a refund here. ^BT

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `ticket_booking_query`, margin 0.1627)  
> Auto resolution: `refund`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: refund. Outcome: agent_answered_unconfirmed. Brand agent turns: 6. Signals seen: refund, information_provided. Evidence: [tweet 64619, refund] "Yes, if you decide not to travel on the effected dates due to the strike a refund can be issued."; [tweet 64608, refund] "We would only issue a refund if customers decide to abandon their journey on the effected dates."; [tweet 64616, refund] "If this is a open return journey you will only receive a refund if the whole journey is abandoned."; [tweet 64618, refund] "If you aren't travelling you can apply for a refund from point of purchase."

Fill in CSV row `case_64621`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 195 · `case_2726558`

**First customer message:** @764613 @VirginTrains They really need to sort this out @VirginTrains

**Conversation**

1. **CUSTOMER** `2726558` 2017-11-20 14:51: @764613 @VirginTrains They really need to sort this out @VirginTrains
2. **AGENT** `2730885` 2017-11-20 15:14: @765599 @764613 We're really sorry to hear about this issue. What type of ticket did you have, Phelan? We do have a e-ticket/m-ticket options for our services so you shouldn't have been charged again if you showed this on your phone ^HP

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `chitchat_non_support` (cluster 0 `chitchat_non_support`; runner-up `customer_service_complaint`, margin 0.0362)  
> Auto resolution: `clarification_requested`, resolved: False, DM redirect: False, outcome `redirected_to_channel`  
> Type: clarification_requested. Outcome: redirected_to_channel. Brand agent turns: 1. Signals seen: clarification_requested. Evidence: [tweet 2730885, clarification_requested] "What type of ticket did you have, Phelan?"

Fill in CSV row `case_2726558`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 196 · `case_1336807`

**First customer message:** @431788 @VirginTrains @431569 T

**Conversation**

1. **CUSTOMER** `1336807` 2017-10-28 07:41: @431788 @VirginTrains @431569 T
2. **CUSTOMER** `1336808` 2017-10-28 07:43: @431788 @VirginTrains @431569 Oops Too late - Virgincare!!

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `unclear_or_media_only` (cluster -1 `unclear_or_media_only`; runner-up `unclear_or_media_only`, margin 0.0518)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `no_response`  
> Brand never replied in the public thread (outcome: no_response).

Fill in CSV row `case_1336807`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 197 · `case_686237`

**First customer message:** So @VirginTrains machine at #Stafford ‘sticks’ and one of our tickets isn’t here (but res and receipt is) Told to speak to staff at #Euston to reprint ticket and greeted with the RUDEST most patronising CSA. Not impressed. At all. Poor.

**Conversation**

1. **CUSTOMER** `686237` 2017-11-23 13:06: So @VirginTrains machine at #Stafford ‘sticks’ and one of our tickets isn’t here (but res and receipt is) Told to speak to staff at #Euston to reprint ticket and greeted with the RUDEST most patronising CSA. Not impressed. At all. Poor.
2. **AGENT** `686235` 2017-11-23 13:25: @283827 Really sorry to hear about your experience, Sarah, have they been able to help you with your issue now? ^HP
3. **CUSTOMER** `686236` 2017-11-23 17:43: @VirginTrains No, not at all. Really annoyed. Sabrina was rude at Euston and apparently only option is to buy another ticket? But we have ALL other elements?! Spoiling a nice mother/daughter day! https://t.co/nytzpxszHc
4. **AGENT** `686238` 2017-11-23 17:48: @283827 Sorry for your experience, Sarah. We'll pass your comments on regarding this situation ^MM

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `ticket_booking_query` (cluster 10 `ticket_booking_query`; runner-up `customer_service_complaint`, margin 0.0056)  
> Auto resolution: `feedback_acknowledged`, resolved: False, DM redirect: False, outcome `agent_answered_unconfirmed`  
> Type: feedback_acknowledged. Outcome: agent_answered_unconfirmed. Brand agent turns: 2. Signals seen: feedback_acknowledged, other. Evidence: [tweet 686238, feedback_acknowledged] "We'll pass your comments on regarding this situation"

Fill in CSV row `case_686237`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 198 · `case_1669638`

**First customer message:** @VirginTrains second journey this weekend and second time I couldn’t get anything from shop due payment issues. Why do I pay £70 for again?

**Conversation**

1. **CUSTOMER** `1669638` 2017-11-06 12:23: @VirginTrains second journey this weekend and second time I couldn’t get anything from shop due payment issues. Why do I pay £70 for again?
2. **AGENT** `1669634` 2017-11-06 12:34: @155058 Sorry to hear this, which service was this on? ^LC
3. **CUSTOMER** `1669636` 2017-11-06 12:39: @VirginTrains 11.50 to Euston. It’s the softwares fault.What can the staff do? Give the whole train free food?!You had issues Fri too, sort your tech out!
4. **CUSTOMER** `1669635` 2017-11-06 12:42: @VirginTrains I have no choice but to use your service too. No choice and the. Your treat is so terribly. Don’t get me started on 1st class. 6 cabins?!
5. **CUSTOMER** `1669637` 2017-11-06 12:44: @VirginTrains Do they ever sell them out?! PFT. https://t.co/3kWDJhGGa9 degrade my journey in some manner&amp;I have to put up with it because who will help?

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `delay_repay_refund_claim` (cluster 3 `delay_repay_refund_claim`; runner-up `ticket_booking_query`, margin 0.0624)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `customer_followup_unanswered`  
> Type: unresolved. Outcome: customer_followup_unanswered. Brand agent turns: 1. Signals seen: clarification_requested. Evidence: [tweet 1669634, clarification_requested] "Sorry to hear this, which service was this on?"

Fill in CSV row `case_1669638`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 199 · `case_1336801`

**First customer message:** @VirginTrains @431569 Not 'high volume of passengers' but reduction in service you provide, I think is the clarity you mean. Adjust tickets?

**Conversation**

1. **CUSTOMER** `1336801` 2017-10-28 11:48: @VirginTrains @431569 Not 'high volume of passengers' but reduction in service you provide, I think is the clarity you mean. Adjust tickets?

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `seat_reservation_issue` (cluster 6 `seat_reservation_issue`; runner-up `ticket_booking_query`, margin 0.0202)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `no_response`  
> Brand never replied in the public thread (outcome: no_response).

Fill in CSV row `case_1336801`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---

### 200 · `case_756985`

**First customer message:** @120576 @VirginTrains @300908 https://t.co/Fm0MsesHyO

**Conversation**

1. **CUSTOMER** `756985` 2017-10-11 19:24: @120576 @VirginTrains @300908 https://t.co/Fm0MsesHyO

> **System suggestion (label the message first; do not anchor on this)**  
> Candidate intent: `unclear_or_media_only` (cluster -1 `unclear_or_media_only`; runner-up `unclear_or_media_only`, margin 0.0518)  
> Auto resolution: `unresolved`, resolved: False, DM redirect: False, outcome `no_response`  
> Brand never replied in the public thread (outcome: no_response).

Fill in CSV row `case_756985`: `human_intent`, `human_resolution_type`, `human_resolved`, `human_escalation_signal`, `human_notes`

---
