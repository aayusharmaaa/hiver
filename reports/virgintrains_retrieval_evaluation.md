# VirginTrains retrieval evaluation (proxy)

> **This is a proxy retrieval evaluation, not human relevance ground truth.** The taxonomy is a *candidate* taxonomy derived from data exploration (`CANDIDATE_NOT_GROUND_TRUTH`); human taxonomy calibration is intentionally not a blocking dependency for this prototype. No golden case is used as a query, as corpus, or for tuning.

## Setup
- **Corpus:** 10,380 historical resolution episodes (`train_retrieval`, strong evidence only).
- **Queries:** 1,712 non-golden `dev_calibration` cases with a strong resolution label and a non-fallback candidate intent (1 dropped for having no relevant case in the corpus). Query text = what the customer said before the first brand reply.
- **Split:** 855 tune / 857 test (stable hash). Weights are chosen on the tune half; **all numbers below are on the test half**.
- **Relevance (headline):** same candidate intent **and** same rule-derived resolution type (avg 651 relevant cases per query). **Secondary:** same candidate intent only (avg 1593).
- **Recall@k** is the hit-rate form: the share of queries with at least one relevant case in the top k.
- **Weights (tuned on the tune half):** semantic weight 0.9, intent bonus 0.15 (a soft additive bonus on the 0-1 normalised score, not a filter).

### Caveats that matter
- The *Hybrid + intent* strategy and the relevance rule share the same candidate taxonomy, so that row is partly self-confirming. See the wrong-intent stress test.
- The resolution type is a heuristic label from the brand's reply, so some results counted as irrelevant may be sensible, and vice versa.
- Cross-split near-duplicates (templated tweets) can inflate scores: 0.0% of test queries have a top-1 hybrid result with identical cleaned text.

## Headline results (test half)
| strategy | R@1 | R@3 | R@5 | MRR | P@5 | MRR 95% CI |
|---|---|---|---|---|---|---|
| BM25 | 0.223 | 0.447 | 0.551 | 0.372 | 0.216 | 0.349-0.396 |
| Embedding | 0.287 | 0.522 | 0.616 | 0.443 | 0.277 | 0.418-0.470 |
| Hybrid | 0.299 | 0.518 | 0.623 | 0.445 | 0.281 | 0.420-0.472 |
| Hybrid + intent | 0.407 | 0.690 | 0.784 | 0.572 | 0.414 | 0.547-0.598 |
| Random (baseline) | 0.053 | 0.156 | 0.237 | 0.152 | 0.061 | - |

**Best by MRR:** Hybrid + intent.

## Secondary relevance: same candidate intent only
| strategy | R@1 | R@3 | R@5 | MRR | P@5 | MRR 95% CI |
|---|---|---|---|---|---|---|
| BM25 | 0.520 | 0.758 | 0.841 | 0.660 | 0.488 | 0.635-0.685 |
| Embedding | 0.672 | 0.866 | 0.907 | 0.776 | 0.639 | 0.754-0.798 |
| Hybrid | 0.669 | 0.851 | 0.907 | 0.769 | 0.630 | 0.747-0.792 |
| Hybrid + intent | 0.953 | 0.991 | 0.995 | 0.973 | 0.947 | 0.964-0.981 |
| Random (baseline) | 0.149 | 0.359 | 0.499 | 0.311 | 0.150 | - |

## Tune half (sanity check that the test half is not unusual)
| strategy | R@1 | R@3 | R@5 | MRR | P@5 | MRR 95% CI |
|---|---|---|---|---|---|---|
| BM25 | 0.253 | 0.451 | 0.560 | 0.391 | 0.224 | 0.364-0.416 |
| Embedding | 0.325 | 0.553 | 0.663 | 0.475 | 0.311 | 0.450-0.501 |
| Hybrid | 0.329 | 0.563 | 0.664 | 0.479 | 0.312 | 0.454-0.505 |
| Hybrid + intent | 0.454 | 0.705 | 0.787 | 0.602 | 0.421 | 0.578-0.627 |

## Weight selection (tune half only)
Semantic weight (clean hybrid MRR):
| sem_weight | mrr |
|---|---|
| 0.000 | 0.391 |
| 0.100 | 0.402 |
| 0.200 | 0.416 |
| 0.300 | 0.428 |
| 0.400 | 0.435 |
| 0.500 | 0.449 |
| 0.600 | 0.453 |
| 0.700 | 0.467 |
| 0.800 | 0.474 |
| 0.900 | 0.479 |
| 1.000 | 0.475 |

Intent bonus (chosen on the mean of clean and 30%-wrong intents, so the weight does not over-trust an imperfect candidate intent):
| intent_weight | mrr_clean_intent | mrr_noisy_intent | objective |
|---|---|---|---|
| 0.000 | 0.479 | 0.479 | 0.479 |
| 0.050 | 0.552 | 0.528 | 0.540 |
| 0.100 | 0.588 | 0.542 | 0.565 |
| 0.150 | 0.602 | 0.536 | 0.569 |
| 0.200 | 0.608 | 0.523 | 0.566 |
| 0.300 | 0.612 | 0.478 | 0.545 |
| 0.500 | 0.613 | 0.430 | 0.521 |

## Stress test: what if the candidate intent is wrong? (test half)
| wrong_intent_rate | mode | recall@5 | mrr |
|---|---|---|---|
| 0.000 | query only | 0.623 | 0.445 |
| 0.000 | soft intent bonus | 0.784 | 0.572 |
| 0.000 | hard intent filter | 0.803 | 0.587 |
| 0.150 | query only | 0.623 | 0.445 |
| 0.150 | soft intent bonus | 0.736 | 0.533 |
| 0.150 | hard intent filter | 0.670 | 0.492 |
| 0.300 | query only | 0.623 | 0.445 |
| 0.300 | soft intent bonus | 0.714 | 0.519 |
| 0.300 | hard intent filter | 0.555 | 0.404 |
| 0.500 | query only | 0.623 | 0.445 |
| 0.500 | soft intent bonus | 0.639 | 0.469 |
| 0.500 | hard intent filter | 0.403 | 0.299 |

## Index-text ablation: index = customer_problem + historical_response (same weights, test half)
| strategy | R@1 | R@3 | R@5 | MRR | P@5 | MRR 95% CI |
|---|---|---|---|---|---|---|
| BM25 | 0.238 | 0.431 | 0.541 | 0.377 | 0.226 | 0.350-0.401 |
| Embedding | 0.278 | 0.508 | 0.636 | 0.437 | 0.282 | 0.412-0.462 |
| Hybrid | 0.291 | 0.527 | 0.636 | 0.443 | 0.285 | 0.417-0.467 |
| Hybrid + intent | 0.438 | 0.693 | 0.782 | 0.590 | 0.425 | 0.565-0.616 |

## Where it works and where it fails (Hybrid + intent, headline relevance)
By candidate intent:
| intent | n | R@1 | R@5 | MRR |
|---|---|---|---|---|
| service_status_delay_enquiry | 268 | 0.485 | 0.836 | 0.636 |
| journey_disruption_complaint | 130 | 0.377 | 0.792 | 0.548 |
| seat_reservation_issue | 74 | 0.473 | 0.757 | 0.602 |
| ticket_booking_query | 72 | 0.431 | 0.806 | 0.590 |
| customer_service_complaint | 67 | 0.224 | 0.657 | 0.413 |
| praise_positive_feedback | 64 | 0.375 | 0.828 | 0.588 |
| delay_repay_refund_claim | 63 | 0.222 | 0.746 | 0.452 |
| chitchat_non_support | 57 | 0.509 | 0.754 | 0.637 |
| onboard_wifi_issue | 35 | 0.400 | 0.714 | 0.538 |
| first_class_catering_issue | 27 | 0.296 | 0.704 | 0.464 |

By query length:
| length | n | R@1 | R@5 | MRR |
|---|---|---|---|---|
| 8-15 tokens | 527 | 0.381 | 0.772 | 0.551 |
| 4-7 tokens | 181 | 0.492 | 0.856 | 0.652 |
| 16+ tokens | 121 | 0.339 | 0.711 | 0.509 |
| 1-3 tokens | 28 | 0.643 | 0.857 | 0.733 |

By the historical agent's resolution type:
| resolution type | n | R@1 | R@5 | MRR |
|---|---|---|---|---|
| information_provided | 471 | 0.597 | 0.958 | 0.750 |
| self_service | 109 | 0.183 | 0.642 | 0.384 |
| compensation | 79 | 0.203 | 0.671 | 0.408 |
| feedback_acknowledged | 67 | 0.104 | 0.463 | 0.265 |
| redirected_to_other_operator | 50 | 0.220 | 0.540 | 0.381 |
| redirected_to_dm | 33 | 0.182 | 0.515 | 0.358 |
| refund | 22 | 0.091 | 0.409 | 0.215 |
| troubleshooting | 14 | 0.286 | 0.714 | 0.418 |
| escalated | 12 | 0.167 | 0.333 | 0.303 |

Queries with no relevant case in the top 5: **185** of 857. Kinds: {'same_intent_but_other_resolution_type': 178, 'very_short_query': 4, 'no_same_intent_in_top5': 3}.

## Good retrieval examples

### QUERY `case_1295259` (candidate intent: service_status_delay_enquiry; agent resolution: compensation; first relevant at rank 1)
> please explain why you are failing to deliver your transportation from London from Euston. 2100 canceled 2140 delayed I need to get back to Manchester

**TOP RETRIEVED CASES**

1. `case_1426856` score 1.10 - service_status_delay_enquiry / compensation - **relevant**
   - problem: 5:20 from Euston to Manchester has been delayed for ages, how do I complain?
   - response: You can claim for the delays here - https://t.co/alwxuSCJ0l
   - why: same candidate intent (service_status_delay_enquiry); same resolution type (compensation); shared terms: delayed, euston, manchester
2. `case_1325187` score 1.10 - service_status_delay_enquiry / self_service - **not relevant**
   - problem: my 12.40 London Euston to Manchester train is cancelled. What do I do?
   - response: We are awaiting an update as to when the line will reopen. Live Updates can be found via https://t.co/QlcukyjBGT
   - why: same candidate intent (service_status_delay_enquiry); different resolution type (self_service vs query's compensation); shared terms: euston, london, manchester
3. `case_1462303` score 1.08 - service_status_delay_enquiry / information_provided - **not relevant**
   - problem: the train to Manchester from Euston at 13:00 is cancelled how do i get there as soon as possible?
   - response: You'd need to get the next available one after your booked one. | We wouldn't know which ones will be free, however Coach C and U will be unreserved.
   - why: same candidate intent (service_status_delay_enquiry); different resolution type (information_provided vs query's compensation); shared terms: euston, manchester

### QUERY `case_676680` (candidate intent: onboard_wifi_issue; agent resolution: troubleshooting; first relevant at rank 3)
> since latest windows update laptop won't connect fully to 1st class Wi-Fi, I keep getting this message when connecting. Any ideas to solve?

**TOP RETRIEVED CASES**

1. `case_813000` score 1.08 - onboard_wifi_issue / self_service - **not relevant**
   - problem: Whenever I️ try to connect to @VirginTrains wifi. #frustrated #firstclass #reallynot
   - response: Try using https://t.co/1JEOSVPtGN
   - why: same candidate intent (onboard_wifi_issue); different resolution type (self_service vs query's troubleshooting); shared terms: connect
2. `case_2671264` score 0.98 - onboard_wifi_issue / information_provided - **not relevant**
   - problem: WiFi never works in first class 😡😡
   - response: I see, might be worth a call to the support team on 0330 088 1271, Paul | Thanks Paul and apologies for this
   - why: same candidate intent (onboard_wifi_issue); different resolution type (information_provided vs query's troubleshooting); shared terms: class
3. `case_614064` score 0.96 - onboard_wifi_issue / troubleshooting - **relevant**
   - problem: do you have a weblink so I can connect to your Wi-fi in first class please? It never connects automatically 🤬
   - response: If you type https://t.co/1JEOSVPtGN into your browser's URL bar, this should load up the WiFi.
   - why: same candidate intent (onboard_wifi_issue); same resolution type (troubleshooting); shared terms: class, connect, fi, wi

### QUERY `case_2515078` (candidate intent: ticket_booking_query; agent resolution: information_provided; first relevant at rank 1)
> > next one with an advance ticket? I booked the whole journey through your website, but the cancelled train is a Northern one.

**TOP RETRIEVED CASES**

1. `case_529445` score 1.03 - ticket_booking_query / information_provided - **relevant**
   - problem: hi. I had a ticket with a booked seat valid for the 8:50 MK to Manchester which has been cancelled. Will I need a new ticket to get a later train?
   - response: Your ticket will be valid on the next available service Marc. Sorry for the cancellation of your train.
   - why: same candidate intent (ticket_booking_query); same resolution type (information_provided); shared terms: booked, cancelled, ticket, train
2. `case_1340118` score 1.02 - ticket_booking_query / information_provided - **relevant**
   - problem: Hi @VirginTrains - I've just got this but no idea if I can use my advance ticket on another train? Please advise.
   - response: Hi there, you can use your ticket on the next available service
   - why: same candidate intent (ticket_booking_query); same resolution type (information_provided); shared terms: advance, ticket, train
3. `case_857033` score 1.01 - ticket_booking_query / information_provided - **relevant**
   - problem: hi. Booked off peak ticket on website and ticket is for a London midland route. Looks like will miss the train due to traffic
   - response: Hi Darren, if you have an Off-Peak ticket then you should be fine to travel on the next O/P service :) | Thanks for confirming Darren. If your ticket is valid on London …
   - why: same candidate intent (ticket_booking_query); same resolution type (information_provided); shared terms: booked, ticket, train, website

### QUERY `case_1639512` (candidate intent: praise_positive_feedback; agent resolution: information_provided; first relevant at rank 2)
> Fab wknd in Wales for nephew's baptism. 2.5 hrs delayed (no fault of @VirginTrains -their staff were excellent) nearly home, but shattered!

**TOP RETRIEVED CASES**

1. `case_1355423` score 1.07 - praise_positive_feedback / feedback_acknowledged - **not relevant**
   - problem: Fab service on the 13.40 Eus-Man. Friendly staff & very relaxing journey. 1st experience of 1st class service. Well worth it!
   - response: Love it, we'll be sure to pass on your kind words, Sharon 😃
   - why: same candidate intent (praise_positive_feedback); different resolution type (feedback_acknowledged vs query's information_provided); shared terms: fab, staff
2. `case_1087069` score 1.06 - praise_positive_feedback / information_provided - **relevant**
   - problem: excellent service Warrington to Edinburgh. Son unfortunately sick in carriage. Guard and staff couldn’t do enough to help
   - response: Oh no 🙈 Hope he's feeling better today :) Thanks for getting in touch
   - why: same candidate intent (praise_positive_feedback); same resolution type (information_provided); shared terms: excellent, staff
3. `case_1748987` score 1.05 - praise_positive_feedback / feedback_acknowledged - **not relevant**
   - problem: Congratulations! @VirginTrains on excellent service 14.15h Birmingham to Preston so superior to @GWRHelp #caringcounts
   - response: Thanks Harry, we'll be sure to pass your comments on 😊
   - why: same candidate intent (praise_positive_feedback); different resolution type (feedback_acknowledged vs query's information_provided); shared terms: excellent

### QUERY `case_2733671` (candidate intent: chitchat_non_support; agent resolution: information_provided; first relevant at rank 1)
> Previous @virgintrains “talking toilet” was merely insipid & mildly aggravating New “cheery American” Christmas version makes me want to commit violence against perpetrator responsible!

**TOP RETRIEVED CASES**

1. `case_147515` score 1.10 - chitchat_non_support / information_provided - **relevant**
   - problem: loving the new talky toilet announcements
   - response: We love it too Ben! Bet you didn't expect to her Will Ferrell whilst in our loos today.
   - why: same candidate intent (chitchat_non_support); same resolution type (information_provided); shared terms: new, toilet
2. `case_1139739` score 1.07 - chitchat_non_support / feedback_acknowledged - **not relevant**
   - problem: Dear @VirginTrains - the will Ferrell toilet announcement thing is irritating and intrusive. Make it stop!
   - response: Well that takes the fun out of it. We'll pass on your comments, Chris
   - why: same candidate intent (chitchat_non_support); different resolution type (feedback_acknowledged vs query's information_provided); shared terms: make, toilet
3. `case_2902920` score 1.04 - chitchat_non_support / information_provided - **relevant**
   - problem: Loving the new ‘toilet voice’!! 😂
   - response: Can't beat a bit of Will Ferrell :)
   - why: same candidate intent (chitchat_non_support); same resolution type (information_provided); shared terms: new, toilet

### QUERY `case_1320725` (candidate intent: seat_reservation_issue; agent resolution: information_provided; first relevant at rank 3)
> Loving the announcements on @VirginTrains 🚂 this morning: “The onboard shop is located in carriage C - that’s C for cheese sandwich.” 😂

**TOP RETRIEVED CASES**

1. `case_2353145` score 1.02 - seat_reservation_issue / self_service - **not relevant**
   - problem: Thank you for being so delayed in changing the carriage letters and seat reservations on the 18:30 from New Street to Euston, I really wanted to play musical chairs and …
   - response: We'll pass on your comments regarding this. If you do wish to make a complaint regarding the staff please get in touch through our website: https://t.co/t20UbyOCSn
   - why: same candidate intent (seat_reservation_issue); different resolution type (self_service vs query's information_provided); shared terms: carriage
2. `case_2663916` score 0.97 - seat_reservation_issue / self_service - **not relevant**
   - problem: yet again battling to get a table seat in stinky carriage C to be able to work.on my journey. This unreserved switch is awful You've basically thrown two fingers up at d…
   - response: Not great to hear, Neil :/ Apologies for the inconvenience caused. You can block book a seat at the ticket office for your
   - why: same candidate intent (seat_reservation_issue); different resolution type (self_service vs query's information_provided); shared terms: c, carriage
3. `case_2879124` score 0.96 - seat_reservation_issue / information_provided - **relevant**
   - problem: Probably the entirety of coach c (not the people. Just the contents of the shop)...
   - response: We're with you on that one! Contemplating ordering a second breakfast..
   - why: same candidate intent (seat_reservation_issue); same resolution type (information_provided); shared terms: c, shop

## Bad retrieval examples

### QUERY `case_1011870` (candidate intent: chitchat_non_support; agent resolution: troubleshooting; first relevant at rank 47)
> Hi @VirginTrains on what planet can you justify this price? Will I have my own private carriage, hand-waited on by Tom Hardy? 🙃

**TOP RETRIEVED CASES**

1. `case_699681` score 0.93 - first_class_catering_issue / information_provided - **not relevant**
   - problem: How does it even cost that much!! I would be expecting a first class carriage for myself with a butler. I could fly to America for that! #MerryChristmas #thingsicouldget…
   - response: We do offer Advance tickets which are subject to availability. If all these Advance fares sell out the only tickets available would be open tickets, which are more expen…
   - why: different candidate intent (first_class_catering_issue vs query chitchat_non_support); different resolution type (information_provided vs query's troubleshooting); shared terms: carriage
2. `case_1438376` score 0.92 - seat_reservation_issue / information_provided - **not relevant**
   - problem: Your carriage awaits - your usual seat? On the floor. Again. Such value for money. @VirginTrains
   - response: As you can travel on any Off-Peak service with your ticket. You would be required to reserve your own seat | once you had chosen which service you were travelling on. We…
   - why: different candidate intent (seat_reservation_issue vs query chitchat_non_support); different resolution type (information_provided vs query's troubleshooting); shared terms: carriage
3. `case_2723805` score 0.81 - chitchat_non_support / information_provided - **not relevant**
   - problem: Hello @VirginTrains myself & buddy's have been sitting on the floor for 4 hours now on the way back from Edinburgh.We spent over £680 on the tickets & we are on your fin…
   - response: You can always book reservations through us. Again we're really sorry about your experience yesterday. | I'm afraid that if you don't reserve a seat and the service is b…
   - why: same candidate intent (chitchat_non_support); different resolution type (information_provided vs query's troubleshooting)

### QUERY `case_2960895` (candidate intent: praise_positive_feedback; agent resolution: self_service; first relevant at rank 9)
> Dear @VirginTrains, whilst I really appreciate your on average 2 hour service for London Euston to Manchester Piccadilly, I would also love for you to sort out your air conditioning/heating machine hybrid you appear to have on full blast on my current £80 per…

**TOP RETRIEVED CASES**

1. `case_2828509` score 1.09 - praise_positive_feedback / feedback_acknowledged - **not relevant**
   - problem: Such a beautiful journey to and from london over the weekend @VirginTrains . Was just a shame about the mix up with prices and payments. But very helpful staff on the 17…
   - response: Thanks for the great feedback. Sorry to hear there were some issues, hope they all got sorted for you
   - why: same candidate intent (praise_positive_feedback); different resolution type (feedback_acknowledged vs query's self_service); shared terms: euston, journey, london, piccadilly
2. `case_1324084` score 1.05 - praise_positive_feedback / information_provided - **not relevant**
   - problem: What's the status of @VirginTrains London Euston-Manchester service? #VTUPDATE anyone at #LondonEuston. About to head there from east london
   - response: These services are currently showing as running, however may meet with delays due to disruption earlier today.
   - why: same candidate intent (praise_positive_feedback); different resolution type (information_provided vs query's self_service); shared terms: euston, london, manchester, service
3. `case_1295472` score 1.04 - praise_positive_feedback / information_provided - **not relevant**
   - problem: Is this effecting London Euston service's heading to Manchester Piccidilly?
   - response: It is, please travel with @122155 from St Pancras to Sheffield and then onwards to Manchester | That service will still be running Joe
   - why: same candidate intent (praise_positive_feedback); different resolution type (information_provided vs query's self_service); shared terms: euston, london, manchester, service

### QUERY `case_527334` (candidate intent: service_status_delay_enquiry; agent resolution: refund; first relevant at rank 246)
> trying to get to Birmingham now can I use virgin from Watford with this??

**TOP RETRIEVED CASES**

1. `case_1295501` score 1.15 - service_status_delay_enquiry / information_provided - **not relevant**
   - problem: Anyway to get to Birmingham from Watford Junction?
   - response: Services are now running so you can wait for the next available service.
   - why: same candidate intent (service_status_delay_enquiry); different resolution type (information_provided vs query's refund); shared terms: birmingham, watford
2. `case_1324152` score 1.02 - service_status_delay_enquiry / information_provided - **not relevant**
   - problem: What’s the easiest way to get from watford junction to Birmingham international / new street / Solihull?
   - response: If you speak to station staff they will be able to advise on the best route Jamie.
   - why: same candidate intent (service_status_delay_enquiry); different resolution type (information_provided vs query's refund); shared terms: birmingham, watford
3. `case_691090` score 1.00 - service_status_delay_enquiry / information_provided - **not relevant**
   - problem: whats the best way of getting back towards Birmingham from Watford Junction? All services look to be cancelled...
   - response: That service will run to Birmingham :-) | Ahhhhhh, it's terminating at Birmingham International, you will be able to connect from there though
   - why: same candidate intent (service_status_delay_enquiry); different resolution type (information_provided vs query's refund); shared terms: birmingham, watford

### QUERY `case_2125846` (candidate intent: journey_disruption_complaint; agent resolution: feedback_acknowledged; first relevant at rank 23)
> next time you’re replacing/refurbishing trains, any chance of more loos? Had to walk the through 4 carriages 😞

**TOP RETRIEVED CASES**

1. `case_2134051` score 1.12 - journey_disruption_complaint / redirected_to_other_operator - **not relevant**
   - problem: Train from Leeds running well over 45 mins late 1st class carriage 100c 😓 no water in the loos worth every penny!! 😡
   - response: Hi Mark, are you travelling with @120576 ?
   - why: same candidate intent (journey_disruption_complaint); different resolution type (redirected_to_other_operator vs query's feedback_acknowledged); shared terms: carriage, loo, train
2. `case_546166` score 1.03 - journey_disruption_complaint / information_provided - **not relevant**
   - problem: Multiple @VirginTrains cancelled and delays but no extra carriages added to deal with the amount of people trying to get on one train?🙄🙄
   - response: Apologies Caz, but due to the disruption across our network, we wouldn't be able to get extra carriages added to a service
   - why: same candidate intent (journey_disruption_complaint); different resolution type (information_provided vs query's feedback_acknowledged); shared terms: carriage, train
3. `case_799150` score 1.01 - journey_disruption_complaint / information_provided - **not relevant**
   - problem: If @140497 are accepting @VirginTrains customers on our route you’d think they’d put on additional carriages.
   - response: Ticket acceptance is in place with Chiltern Railways.
   - why: same candidate intent (journey_disruption_complaint); different resolution type (information_provided vs query's feedback_acknowledged); shared terms: carriage

### QUERY `case_2581849` (candidate intent: customer_service_complaint; agent resolution: compensation; first relevant at rank 38)
> I just parked for 26 minutes and had to pay £5 - no staff around to sort this out. How do I get my £5 back @VirginTrains ? Mrs Grumpy in Wolverhampton

**TOP RETRIEVED CASES**

1. `case_1301999` score 0.98 - customer_service_complaint / redirected_to_dm - **not relevant**
   - problem: NHS midwife with a rare day off had my wallet stolen. Contacted @VirginTrains to replace my tickets, informed i have to pay again #helpme
   - response: Do you have a crime reference number? Please DM us the details? | We'll respond to your DM
   - why: same candidate intent (customer_service_complaint); different resolution type (redirected_to_dm vs query's compensation); shared terms: pay
2. `case_102438` score 0.98 - customer_service_complaint / self_service - **not relevant**
   - problem: Seriously, my wife just drove to the pickup area in Weston Carpark Crewe and despite a 20 minute free wait and only being there 9 minutes it’s charged us £12?! WTF?
   - response: Apologies Ian, as this is a station issue, you would need to dispute this at the ticket office so this can be looked into further for you
   - why: same candidate intent (customer_service_complaint); different resolution type (self_service vs query's compensation); shared terms: just, minute
3. `case_2418470` score 0.95 - customer_service_complaint / redirected_to_other_operator - **not relevant**
   - problem: poor excuse of a company. Fuming with having to pay an extra £47.80 today after an awful journey to Leeds on 27/10/17 😡😡😡
   - response: You'll need to speak with @120576 as they're the team they can advise on those services
   - why: same candidate intent (customer_service_complaint); different resolution type (redirected_to_other_operator vs query's compensation); shared terms: pay

### QUERY `case_2472316` (candidate intent: seat_reservation_issue; agent resolution: self_service; first relevant at rank 12)
> Dog sat on table in First class? Sod any paying passengers who are allergic to them

**TOP RETRIEVED CASES**

1. `case_628955` score 1.00 - chitchat_non_support / information_provided - **not relevant**
   - problem: what is your policy on dogs? I'm allergic so sat in drinks carriage c and there is a dog!
   - response: Dogs are allowed on our services, if you speak with a member of the onboard team they can advise on available seats in other carriages if you wish to move seats. | They …
   - why: different candidate intent (chitchat_non_support vs query seat_reservation_issue); different resolution type (information_provided vs query's self_service); shared terms: allergic, dog, sat
2. `case_999463` score 0.95 - seat_reservation_issue / information_provided - **not relevant**
   - problem: Do u need to buy a ticket for a dog/Cat if occupying a seat?
   - response: They don't need tickets, but we do ask that the don't sit on the seats unless it's covered.
   - why: same candidate intent (seat_reservation_issue); different resolution type (information_provided vs query's self_service); shared terms: dog
3. `case_1163039` score 0.88 - seat_reservation_issue / feedback_acknowledged - **not relevant**
   - problem: Well done! Booked Priority Seat as we have a Guide Dog. Why did you give us standard seats? Pointless phoning up to book!
   - response: We'll pass on your comments
   - why: same candidate intent (seat_reservation_issue); different resolution type (feedback_acknowledged vs query's self_service); shared terms: dog

