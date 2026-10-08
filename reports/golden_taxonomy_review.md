# Golden set: taxonomy review

All 250 golden cases have final human-reviewed labels: 100 were labelled blind by a human, the rest began as assistant
drafts that a human reviewed (see the README's provenance note). This compares the candidate taxonomy's cluster-derived
intent with the final `gold_intent`, to decide whether the taxonomy should change before it is frozen. It is **not** an evaluation of the agent and
reports no accuracy figures. Nothing here was applied automatically.

- Taxonomy reference: candidate registry, status `CANDIDATE_NOT_GROUND_TRUTH`. (Candidate taxonomy — provisional; choose the best-supported intent. You may use NEW:<snake_case> if none fits.)
- Golden cases were sampled with stratification, so these counts are not natural prevalence (see `golden_stratum_weight`).
- The candidate intent was never shown to the labeler; it is joined here only after labeling finished.
- If the taxonomy changes, the gold labels stay as the human entered them; any re-mapping must be explicit and reviewed.

## Intent counts

| intent | candidate | gold |
|---|---|---|
| `chitchat_non_support` | 21 | 11 |
| `customer_service_complaint` | 20 | 19 |
| `delay_repay_refund_claim` | 20 | 29 |
| `first_class_catering_issue` | 15 | 7 |
| `journey_disruption_complaint` | 28 | 37 |
| `onboard_wifi_issue` | 15 | 16 |
| `praise_positive_feedback` | 25 | 40 |
| `seat_reservation_issue` | 21 | 16 |
| `service_status_delay_enquiry` | 62 | 31 |
| `ticket_booking_query` | 21 | 35 |
| `unclear_or_media_only` | 2 | 7 |
| `NEW:lost_property` | 0 | 2 |

## Candidate vs gold disagreement

Gold differs from the candidate intent in 131 of 250 cases.

| candidate intent | cases | gold same | gold different | most common gold instead |
|---|---|---|---|---|
| `chitchat_non_support` | 21 | 5 | 16 | journey_disruption_complaint (6), customer_service_complaint (4), unclear_or_media_only (3) |
| `customer_service_complaint` | 20 | 9 | 11 | praise_positive_feedback (5), delay_repay_refund_claim (2), ticket_booking_query (2) |
| `delay_repay_refund_claim` | 20 | 17 | 3 | praise_positive_feedback (2), ticket_booking_query (1) |
| `first_class_catering_issue` | 15 | 4 | 11 | ticket_booking_query (4), praise_positive_feedback (2), chitchat_non_support (2) |
| `journey_disruption_complaint` | 28 | 8 | 20 | praise_positive_feedback (6), customer_service_complaint (3), ticket_booking_query (3) |
| `onboard_wifi_issue` | 15 | 13 | 2 | delay_repay_refund_claim (1), journey_disruption_complaint (1) |
| `praise_positive_feedback` | 25 | 16 | 9 | journey_disruption_complaint (3), first_class_catering_issue (1), delay_repay_refund_claim (1) |
| `seat_reservation_issue` | 21 | 8 | 13 | journey_disruption_complaint (7), service_status_delay_enquiry (3), ticket_booking_query (2) |
| `service_status_delay_enquiry` | 62 | 25 | 37 | journey_disruption_complaint (11), ticket_booking_query (9), praise_positive_feedback (6) |
| `ticket_booking_query` | 21 | 13 | 8 | delay_repay_refund_claim (3), seat_reservation_issue (2), customer_service_complaint (1) |
| `unclear_or_media_only` | 2 | 1 | 1 | praise_positive_feedback (1) |

## NEW intents proposed by the labeler

### `NEW:lost_property` (2 cases)

- case_1309039 (candidate `service_status_delay_enquiry`): Lost phone left on train; pointed to lost property line.  
  > @VirginTrains My iPhone is currently taking a train trip without me... I've tracked it from BHI to EUS and back. How can I get it home?
- case_512244 (candidate `praise_positive_feedback`): Wallet left on train; pointed to lost property line.  
  > @VirginTrains Help me Vir-Gin-Wan Kenobi, you're my only hope. Friend arrived at Manchester Picadilly around 9.10pm; left wallet in an M&am…


## Major confusion pairs

Unordered pairs of different intents, counted over cases where candidate and gold disagree.

| intent A | intent B | cases |
|---|---|---|
| `journey_disruption_complaint` | `service_status_delay_enquiry` | 13 |
| `service_status_delay_enquiry` | `ticket_booking_query` | 10 |
| `journey_disruption_complaint` | `praise_positive_feedback` | 9 |
| `journey_disruption_complaint` | `seat_reservation_issue` | 9 |
| `praise_positive_feedback` | `service_status_delay_enquiry` | 6 |
| `chitchat_non_support` | `journey_disruption_complaint` | 6 |
| `customer_service_complaint` | `praise_positive_feedback` | 5 |
| `seat_reservation_issue` | `service_status_delay_enquiry` | 5 |
| `chitchat_non_support` | `customer_service_complaint` | 4 |
| `seat_reservation_issue` | `ticket_booking_query` | 4 |
| `delay_repay_refund_claim` | `ticket_booking_query` | 4 |
| `first_class_catering_issue` | `ticket_booking_query` | 4 |
| `delay_repay_refund_claim` | `service_status_delay_enquiry` | 3 |
| `chitchat_non_support` | `unclear_or_media_only` | 3 |
| `customer_service_complaint` | `ticket_booking_query` | 3 |

## Difficult boundary examples

Disagreements the labeler marked as low or medium confidence, then other disagreements.

| case | candidate | gold | confidence | opening message | notes |
|---|---|---|---|---|---|
| case_1255846 | praise_positive_feedback | ticket_booking_query | low | Thankyou @VirginTrains very nice doing business with ya...... only wanted off at Preston 🖕🏼 https://t.co/l77CTbAZQE | Unclear complaint about missing stop/open return. |
| case_1573957 | service_status_delay_enquiry | journey_disruption_complaint | low | Why does Stockport station have 5x the bike parking of Manchester's main stations? Piccadilly was just redeveloped! #wtf @VirginTrains? htt… | Station facilities (bike parking) feedback; no intent fits well. |
| case_587670 | service_status_delay_enquiry | customer_service_complaint | low | @VirginTrains Can you put live bus times in your stations so its easier for passengers to know their wait for their buses (Crewe especially) | Station improvement suggestion; no intent fits well. |
| case_1035355 | seat_reservation_issue | journey_disruption_complaint | medium | @VirginTrains book a seat at table and plug and plug doesn’t work, cheers!!!! | Seat power socket not working. |
| case_1234365 | service_status_delay_enquiry | ticket_booking_query | medium | @VirginTrains And finally get a Virgin train to EUS. Can you confirm I can travel tomorrow and if so what to do with the app? | - |
| case_1325195 | service_status_delay_enquiry | journey_disruption_complaint | medium | @VirginTrains any idea how long the Euston line will be blocked for? | - |
| case_1343962 | chitchat_non_support | unclear_or_media_only | medium | @VirginTrains can you help? X https://t.co/abcXynNhtz | Question is only in an image; reply suggests an upgrade query. |
| case_136342 | chitchat_non_support | journey_disruption_complaint | medium | @VirginTrains Your cleaner at Stockport is psychic. In the door, sign the sheet and straight back out again. https://t.co/ZauWxsyNXR | Station cleaning complaint. |
| case_1364937 | ticket_booking_query | delay_repay_refund_claim | medium | @VirginTrains why pay for ALL DAY first class return tickets when you don’t run after a certain time and making me pay for more tickets?How? | - |
| case_1506857 | first_class_catering_issue | ticket_booking_query | medium | @VirginTrains hi can you tell me how much it would cost to upgrade to first class on the train on day of travel? | First class upgrade price needs booking reference via DM. |
| case_1563026 | onboard_wifi_issue | journey_disruption_complaint | medium | @VirginTrains why is my prebooked 10:21 train from Kings Cross actually an East Midlands train..without the free wifi i'd purposely booked? | Train swapped to one without wifi; other operator. |
| case_1648575 | journey_disruption_complaint | praise_positive_feedback | medium | @VirginTrains cancelled.Followed tannoy advice &amp; took alt. route via https://t.co/t4QwLsgRcg on both trains;only 1hr late home👍#nocompl… | - |
| case_166715 | chitchat_non_support | unclear_or_media_only | medium | Nothing quite as #discombobulating as hearing #WillFerrell's voice as you sit down on a @VirginTrains #train #toilet. #TGIF #DaddysHome2 | - |
| case_1750708 | service_status_delay_enquiry | ticket_booking_query | medium | @VirginTrains how do you recommend getting from Euston to Lime Street? | Route advice between stations. |
| case_1845809 | journey_disruption_complaint | unclear_or_media_only | medium | @VirginTrains your trains are disgusting.... https://t.co/vnZr1Pf8bH | - |
| case_1854626 | seat_reservation_issue | ticket_booking_query | medium | Story of my trip to Edinburgh with @virgintrains 1. Struggled to remove the added in error ticket from the basket. When I thought I’d sorte… | - |
| case_2106621 | chitchat_non_support | first_class_catering_issue | medium | @VirginTrains cup of tea around £3 but the assistant tells me ‘we don’t have fresh milk spare for you only those shit cartons’!!!! 😕 | Standard-class catering complaint on another operator's service. |
| case_2108252 | journey_disruption_complaint | ticket_booking_query | medium | Sucks that my 16-25 railcard that has saved me sooo much over the years on my travel is coming to an end. How will I afford trains now @Vir… | - |
| case_2129830 | service_status_delay_enquiry | journey_disruption_complaint | medium | Now I’ve got #280characters probably a good time to continue my story. Also delayed massively by @VirginTrains so let’s be honest I have so… | Delay on another operator's service. |
| case_2196804 | chitchat_non_support | unclear_or_media_only | medium | @VirginTrains what is the meaning of this ? https://t.co/E42XIGly4V | Customer provides only a link and asks what it means; the specific issue cannot be determined from the message. |

## Merge / split / rename signals

Heuristics for a human decision (merge: confused both ways in at least 3 cases; split: at least 3 gold intents with none above 60%; rename: at least 50% relabelled to one other intent; add: a NEW intent with at least 3 cases).

- **merge?** `chitchat_non_support` and `first_class_catering_issue` are confused in both directions (1 candidate chitchat_non_support → gold first_class_catering_issue, 2 the other way).
- **merge?** `customer_service_complaint` and `ticket_booking_query` are confused in both directions (2 candidate customer_service_complaint → gold ticket_booking_query, 1 the other way).
- **merge?** `delay_repay_refund_claim` and `praise_positive_feedback` are confused in both directions (2 candidate delay_repay_refund_claim → gold praise_positive_feedback, 1 the other way).
- **merge?** `delay_repay_refund_claim` and `ticket_booking_query` are confused in both directions (1 candidate delay_repay_refund_claim → gold ticket_booking_query, 3 the other way).
- **merge?** `first_class_catering_issue` and `praise_positive_feedback` are confused in both directions (2 candidate first_class_catering_issue → gold praise_positive_feedback, 1 the other way).
- **merge?** `journey_disruption_complaint` and `praise_positive_feedback` are confused in both directions (6 candidate journey_disruption_complaint → gold praise_positive_feedback, 3 the other way).
- **merge?** `journey_disruption_complaint` and `seat_reservation_issue` are confused in both directions (2 candidate journey_disruption_complaint → gold seat_reservation_issue, 7 the other way).
- **merge?** `journey_disruption_complaint` and `service_status_delay_enquiry` are confused in both directions (2 candidate journey_disruption_complaint → gold service_status_delay_enquiry, 11 the other way).
- **merge?** `seat_reservation_issue` and `service_status_delay_enquiry` are confused in both directions (3 candidate seat_reservation_issue → gold service_status_delay_enquiry, 2 the other way).
- **merge?** `seat_reservation_issue` and `ticket_booking_query` are confused in both directions (2 candidate seat_reservation_issue → gold ticket_booking_query, 2 the other way).
- **merge?** `service_status_delay_enquiry` and `ticket_booking_query` are confused in both directions (9 candidate service_status_delay_enquiry → gold ticket_booking_query, 1 the other way).
- **split / redefine?** candidate `chitchat_non_support` (21 cases) spreads over 7 gold intents: journey_disruption_complaint 6, chitchat_non_support 5, customer_service_complaint 4, unclear_or_media_only 3.
- **split / redefine?** candidate `customer_service_complaint` (20 cases) spreads over 6 gold intents: customer_service_complaint 9, praise_positive_feedback 5, delay_repay_refund_claim 2, ticket_booking_query 2.
- **split / redefine?** candidate `first_class_catering_issue` (15 cases) spreads over 7 gold intents: first_class_catering_issue 4, ticket_booking_query 4, praise_positive_feedback 2, chitchat_non_support 2.
- **split / redefine?** candidate `journey_disruption_complaint` (28 cases) spreads over 9 gold intents: journey_disruption_complaint 8, praise_positive_feedback 6, customer_service_complaint 3, ticket_booking_query 3.
- **split / redefine?** candidate `seat_reservation_issue` (21 cases) spreads over 5 gold intents: seat_reservation_issue 8, journey_disruption_complaint 7, service_status_delay_enquiry 3, ticket_booking_query 2.
- **split / redefine?** candidate `service_status_delay_enquiry` (62 cases) spreads over 11 gold intents: service_status_delay_enquiry 25, journey_disruption_complaint 11, ticket_booking_query 9, praise_positive_feedback 6.
- **rename / redefine?** 1 of 2 cases of candidate `unclear_or_media_only` were labelled `praise_positive_feedback`.
