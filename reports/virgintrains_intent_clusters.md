# VirginTrains candidate intent clusters

Embedding model `sentence-transformers/all-MiniLM-L6-v2`, KMeans k=12, fit on 15,528 train+dev openers (entities masked). TF-IDF/LSA vs embedding agreement: NMI 0.28, ARI 0.17.
These are **candidate** clusters; names and descriptions are hand-written after reading examples. Examples come from the train split.

| id | proposed intent | cases | % resolved | % DM | final intent |
|---|---|---:|---:|---:|---|
| 5 | journey_disruption_complaint | 2,093 | 13.2% | 1.9% | journey_disruption_complaint |
| 0 | chitchat_non_support | 2,047 | 10.5% | 1.2% | chitchat_non_support |
| 1 | service_running_enquiry | 1,910 | 19.6% | 2.0% | service_status_delay_enquiry |
| 8 | praise_positive_feedback | 1,481 | 27.8% | 0.3% | praise_positive_feedback |
| 10 | ticket_booking_query | 1,360 | 18.7% | 8.3% | ticket_booking_query |
| 9 | service_status_update_request | 1,309 | 18.2% | 0.9% | service_status_delay_enquiry |
| 6 | seat_reservation_issue | 1,096 | 13.6% | 5.3% | seat_reservation_issue |
| 7 | customer_service_complaint | 1,090 | 10.9% | 10.3% | customer_service_complaint |
| 2 | delay_in_progress_report | 1,032 | 15.5% | 1.8% | service_status_delay_enquiry |
| 3 | delay_repay_refund_claim | 884 | 16.1% | 17.6% | delay_repay_refund_claim |
| 4 | first_class_service_issue | 707 | 18.1% | 1.6% | first_class_catering_issue |
| 11 | onboard_wifi_issue | 519 | 12.9% | 2.5% | onboard_wifi_issue |

## Cluster 5: journey_disruption_complaint

Venting or reporting a disrupted or overcrowded journey (cancellations, half-length trains, announcements, train manager), without a clear question; includes some lost-property and misc. requests.

- Cases: 2,093 (13.5% of clustered); resolved 13.2%; DM redirect 1.9%
- Resolution types: information_provided 39%, unresolved 13%, feedback_acknowledged 9%, redirected_to_other_operator 8%, self_service 8%, compensation 6%
- Confusable with: 1 service_running_enquiry (45%), 8 praise_positive_feedback (16%), 6 seat_reservation_issue (9%)
- Auto-handle hint: `unclear` (NEEDS_REVIEW)

Representative examples:
  - `case_794832` The man loves #Trains too much! @VirginTrains service from Wolves to London in 2 mins anyone?? https://t.co/SGHbuj0lCm
  - `case_2224699` Welcome to this @virgintrains service to Euston. We only have half a train today as the other half decided it didn’t want to go to Wrexham. An unusual problem..
  - `case_98360` @VirginTrains awful that you haven’t forewarned passengers of cancelled trains today! Additional changes and stops for the Liverpool train to get to Holyhead!!!
  - `case_506355` @VirginTrains Seriously what the hell is happening with trains from #euston this evening. #virgintrains
  - `case_1125668` @VirginTrains Train manager ‘Mark’ on Leeds-Kings Cross 13.45 just kicked me off empty carriage. Last time I’ll take a virgin train #thanks

Ambiguous examples:
  - `case_2515069` #GrandCentral train cancelled - not allowed to go on empty @VirginTrains 8:02 but have to stand for two hours on packed 8:31.  _(also close to 1:service_running_enquiry)_
  - `case_1723503` @VirginTrains poor show today #FirstClass Cov to Brum machine unstocked in longe then REFUSED coffee on train #Whatamipayingfor  _(also close to 4:first_class_service_issue)_
  - `case_2639912` @VirginTrains your guy John running the food trolly on the 19.00 lol kings x to Edinburgh is a legend! He makes train journeys better! Give that man a raise!  _(also close to 8:praise_positive_feedback)_
  - `case_1271503` @VirginTrains Sandra the train manager on the 10.50 new street- London was amazing  _(also close to 8:praise_positive_feedback)_
  - `case_1091025` @VirginTrains it’s well hot on your carriages, please can you turn the heating off. 17.40 Euston to Manchester Piccadilly. Thanks!  _(also close to 8:praise_positive_feedback)_

Representative resolution patterns (first agent reply):
  - 12x "Which service are you on?"
  - 10x "If you mention this to the onboard team they'd look to alter this for you"
  - 9x "Which service are you on this afternoon?"
  - 8x "Which service are you on today?"
  - 8x "Sorry to hear this. Which service is this?"

## Cluster 0: chitchat_non_support

Greetings, jokes, photos, and general chatter that do not request support; also contains a few real complaints that were pulled in by short, low-content wording.

- Cases: 2,047 (13.2% of clustered); resolved 10.5%; DM redirect 1.2%
- Resolution types: information_provided 39%, unresolved 26%, other 14%, self_service 7%, feedback_acknowledged 5%, clarification_requested 4%
- Confusable with: 8 praise_positive_feedback (44%), 7 customer_service_complaint (30%), 2 delay_in_progress_report (12%)
- Auto-handle hint: `likely_yes` (NEEDS_REVIEW)

Representative examples:
  - `case_2723805` Hello @VirginTrains myself &amp; buddy's have been sitting on the floor for 4 hours now on the way back from Edinburgh.We spent over £680 on the tickets &amp; we are on your finest carpet next too a smelly,out of order toilet. I'm baffed. …
  - `case_2089364` @120244 @VirginTrains @120576 @121428 @121516 @124107 @LondonMidland Morning everyone! BL
  - `case_1420976` @VirginTrains Morning lovely tweeps! Have a Maccalicious Thursday! X https://t.co/OkcK5lfgmJ
  - `case_2811553` Toilet humour #WillFerrell @VirginTrains made my morning! 😂
  - `case_1397470` Time for work hello @VirginTrains @120576 https://t.co/vz8grBlXFu

Ambiguous examples:
  - `case_2935606` @VirginTrains you know that advert where the girl arrives calm collected? How about showing a more realistic one next time with people standing in the aisles from Crewe to London? I estimate at least 50 standing like me. #ripoff #virgin  _(also close to 8:praise_positive_feedback)_
  - `case_494597` This rather busy and delayed @virgintrains trip would be fine if the wife would stop moaning #doghouse  _(also close to 8:praise_positive_feedback)_
  - `case_2489573` @VirginTrains @425482 Or when we’re likely to depart.  _(also close to 2:delay_in_progress_report)_
  - `case_1609645` @126035 @VirginTrains The Livery is to plain. https://t.co/A8iaYhpVF5  _(also close to 8:praise_positive_feedback)_
  - `case_2689093` Who goes to London to buy trainers? Me. Who goes to London to buy trainers and loses their wallet? Me.  _(also close to 8:praise_positive_feedback)_

Representative resolution patterns (first agent reply):
  - 10x "https://t.co/TK8CcWEeLQ"
  - 4x "😂😂😂 ^MM https://t.co/xN4p7nYwRL"
  - 4x "Which service are you on?"
  - 4x "Oh no, sorry to hear this, which service are you on today please?"
  - 4x "Anything we can help with?"

## Cluster 1: service_running_enquiry

Customer asks whether a specific service is running, cancelled, or when trains will operate (often at a named time or route).

- Cases: 1,910 (12.3% of clustered); resolved 19.6%; DM redirect 2.0%
- Resolution types: information_provided 57%, self_service 14%, compensation 6%, other 5%, unresolved 4%, redirected_to_other_operator 4%
- Confusable with: 5 journey_disruption_complaint (62%), 2 delay_in_progress_report (10%), 10 ticket_booking_query (9%)
- Auto-handle hint: `unclear` (NEEDS_REVIEW)

Representative examples:
  - `case_1637936` @116961 trains Is the 20.55 from Manchester Piccadilly to Euston going anytime soon? Literally no info available at the station
  - `case_1353047` @VirginTrains we are trying to get to manchester this evening from london euston - will trains be running later this evening?
  - `case_1614409` @VirginTrains traveling to Preston from Euston at 1340, please can I have an update on my train??
  - `case_595414` @VirginTrains will trains between euston and manchester be operating tomorrow at 7:30?
  - `case_755645` @VirginTrains my train 19.15 from Bham New St to Lancaster is cancelled. I've been told to get the 19.36 to Crewe..what then?

Ambiguous examples:
  - `case_1329607` @VirginTrains Do we get our ticket fare back???? I’m having to stay an extra night in London can’t even move in Euston let alone get a train if there was to be one!!!  _(also close to 10:ticket_booking_query)_
  - `case_2761567` Well that makes sense @115793 putting a Freight train in front of our @VirginTrains service this morning. Having a rather slow trip to London via Northampton.  _(also close to 5:journey_disruption_complaint)_
  - `case_659091` Missing my @VirginTrains train back north becuase the Northern Line locked us underground for 20 minutes &amp; having to pay 3 times price for a new ticket + an hours wait + 2am arrival is exactly what @22693 was made for.  _(also close to 5:journey_disruption_complaint)_
  - `case_875325` @VirginTrains at this rate it will take you as long as the train was delayed to answer the phone! #hangingonthetelephone  _(also close to 5:journey_disruption_complaint)_
  - `case_1637278` @VirginTrains I am extremely disappointed that I have been waiting for the 21:05 train due to the lateness of YOUR member of staff (1)  _(also close to 5:journey_disruption_complaint)_

Representative resolution patterns (first agent reply):
  - 13x "We are awaiting an update as to when the line will reopen. Live Updates can be found via https://t.co/uVZc5Qy88P"
  - 12x "We're awaiting an update as to when the lines will reopen. Live Updates can be found via https://t.co/Fv5XkzOEzD"
  - 10x "This service is showing to run, however may be met with delay. Please keep up to date here https://t.co/4112Xriq4q"
  - 9x "We don't have an update on this service at the moment, Peter. The staff onboard would be best to advise further"
  - 8x "Which service are you travelling on?"

## Cluster 8: praise_positive_feedback

Thanks, compliments for staff or journeys, and positive feedback.

- Cases: 1,481 (9.5% of clustered); resolved 27.8%; DM redirect 0.3%
- Resolution types: information_provided 42%, other 16%, feedback_acknowledged 13%, unresolved 8%, self_service 7%, redirected_to_other_operator 6%
- Confusable with: 0 chitchat_non_support (36%), 9 service_status_update_request (14%), 5 journey_disruption_complaint (12%)
- Auto-handle hint: `likely_yes` (NEEDS_REVIEW)

Representative examples:
  - `case_701146` Big thanks @VirginTrains for eventually getting us home to Glasgow in the early hours after the flooding in Lancashire last night, it was a bit touch &amp; go for a while, especially in Lancaster. Glasgow crew went above and beyond 👏👏👏👏
  - `case_1127597` Another great @VirginTrains journey to #Peterborough for meeting @385847 -very handy for station 😀 https://t.co/q79A2pr8hz
  - `case_1722568` @nationalrailenq a big thank you to all the staff at #WBQ Station yesterday who were so helpful with travel advice 👏🏼 Got to London on time!
  - `case_557576` Visited #London today and had a fantastic day at @25429 🎅🏼🎄❄️☃️⭐️ Just wanted to give a shout out to @VirginTrains staff at Euston who (despite the chaos...overhead lines...delays) were fantastic and so helpful 👍🏼thank you!
  - `case_2614744` Great journey on @VirginTrains 12.43 from Euston to Wolverhampton. Great service love the 1st class travel ☺

Ambiguous examples:
  - `case_2764863` When you've been up since stupid o'clock without time to grab a brew or get cash out &amp; @virgintrains announce a cash only trolley service. https://t.co/f132XQah1B  _(also close to 5:journey_disruption_complaint)_
  - `case_893251` @VirginTrains WE NEED YOUR HELP! My boyfriend has just left his suit on the train and we have a wedding tomorrow morning. Can you help us?  _(also close to 0:chitchat_non_support)_
  - `case_1327273` @VirginTrains I'm due to get 20:07 Euston to Liv Lime St. What are the chances that'll be affected by today's events? Thanks  _(also close to 2:delay_in_progress_report)_
  - `case_741324` Thanks for kicking me off a nice warm @VirginTrains. 12.10 from Euston cancelled due to crew shortage 20 mins from final dest #pisstake  _(also close to 2:delay_in_progress_report)_
  - `case_552995` @VirginTrains do you think travel tomorrow will be affected?  _(also close to 2:delay_in_progress_report)_

Representative resolution patterns (first agent reply):
  - 19x "Have a great journey!"
  - 6x "Great picture 😃"
  - 6x "Thanks for your comments, we'll pass this onto @120576"
  - 6x "This is one for @120576"
  - 4x "If you speak to staff, they can adjust it for you."

## Cluster 10: ticket_booking_query

Questions about ticket validity, advance tickets, bookings, using a ticket on another train, or app/booking problems.

- Cases: 1,360 (8.8% of clustered); resolved 18.7%; DM redirect 8.3%
- Resolution types: information_provided 47%, self_service 16%, unresolved 8%, redirected_to_dm 7%, other 6%, clarification_requested 4%
- Confusable with: 1 service_running_enquiry (17%), 7 customer_service_complaint (16%), 6 seat_reservation_issue (14%)
- Auto-handle hint: `unclear` (NEEDS_REVIEW)

Representative examples:
  - `case_1759491` @VirginTrains Are advance tickets available for 25 Jan Liverpool to London Euston return?
  - `case_2533670` @VirginTrains this ticket can you got via Birmingham ? https://t.co/9nMtxz4rVD
  - `case_1328100` @VirginTrains Hi, I need help regarding a ticket booking.
  - `case_1087998` @VirginTrains FYI your advance ticket calendar needs updating. Says tickets for 28/12 aren’t avail but when I go to book they are
  - `case_580361` @VirginTrains Sorry to bother you on such a hectic day. I have ADVANCE SINGLE tickets for 14.45 from Wolverhampton to Euston today. Will they be accepted in the morning?

Ambiguous examples:
  - `case_2669898` Thanks a lot @VirginTrains! Mislead advise has meant I’ve checked out of my hotel room early, but now I’m stuck at Euston for 4 hours! Disgusting! Apparently Advance Off-Peak is not the same as an Advance ticket during Off-Peak! @306277 @4…  _(also close to 1:service_running_enquiry)_
  - `case_1329593` @VirginTrains What if you have an off peak return and are trying to travel tonight?  _(also close to 2:delay_in_progress_report)_
  - `case_590467` @VirginTrains hi just wondered if my train ticket will be valid to travel tomorrow instead of today due to overcrowding? Really don't fancy standing for 3 hours from oxenholme to London  _(also close to 1:service_running_enquiry)_
  - `case_81145` @VirginTrains Going to London ex Preston Sunday. Website says plenty parking avail. At Preston but none avail for prebook. Can you pls advise?  _(also close to 6:seat_reservation_issue)_
  - `case_677942` Super happy ticket seller at Rugby station today. Made me smile! Thank you :-D @VirginTrains  _(also close to 8:praise_positive_feedback)_

Representative resolution patterns (first agent reply):
  - 7x "Can you DM us your booking or collection reference number please?"
  - 7x "The Aftersales team would be best to advise on 0344 556 5650"
  - 5x "Yes, you can"
  - 4x "We do have ticket acceptance in place should you choose to travel via our routes"
  - 4x "Yes."

## Cluster 9: service_status_update_request

Short requests for an update on a named service ("what's happening with the 19:37?", "any update?").

- Cases: 1,309 (8.4% of clustered); resolved 18.2%; DM redirect 0.9%
- Resolution types: information_provided 59%, self_service 15%, other 7%, compensation 5%, unresolved 5%, redirected_to_other_operator 3%
- Confusable with: 2 delay_in_progress_report (46%), 8 praise_positive_feedback (34%), 1 service_running_enquiry (7%)
- Auto-handle hint: `unclear` (NEEDS_REVIEW)

Representative examples:
  - `case_1635365` @VirginTrains what’s happening with the 19:37 Euston to Manchester ?
  - `case_544878` @VirginTrains where's the 16:47 from lime street to Euston?
  - `case_494603` @VirginTrains any update on the 17.40 from Euston to Manchester Piccadilly?
  - `case_1327285` Is the 14:07 still going to Liverpool lime street ? @VirginTrains @213768
  - `case_1611326` @VirginTrains 10.38 from Liverpool to Euston not going anywhere. Could you give me an update please?

Ambiguous examples:
  - `case_2561274` @VirginTrains sat on the 17.43 from Euston... when will it depart? Why are there never any announcements?  _(also close to 2:delay_in_progress_report)_
  - `case_749145` @VirginTrains Any news on the 18:52 from Edinburgh? My daughter waiting to come home to Wigan.  _(also close to 2:delay_in_progress_report)_
  - `case_1332127` @virgintrains is the 13:25 Wigan North Western to Glasgow Central likely to get there? No helpful information at the station.  _(also close to 1:service_running_enquiry)_
  - `case_1226147` @VirginTrains when is the next expected departure from London Euston to Manchester? My 1940 services delayed.  _(also close to 2:delay_in_progress_report)_
  - `case_2228246` Steve on the 10.31 @VirginTrains to Milton Keynes...you're a jem!  _(also close to 8:praise_positive_feedback)_

Representative resolution patterns (first agent reply):
  - 11x "We are awaiting an update as to when the line will reopen. Live Updates can be found via https://t.co/QlcukyjBGT"
  - 9x "We don't have an update on this service, Keeley. The train manager would be best to advise further"
  - 8x "We're awaiting an update as to when the lines will reopen. Live Updates can be found via https://t.co/yCdNlgQWid"
  - 6x "Which service are you on?"
  - 6x "This service is showing to run Sarah. You can keep up to date with this service here https://t.co/DamnE1x0EC"

## Cluster 6: seat_reservation_issue

Reserved seats missing, double-booked, or not honoured; overcrowding and people standing.

- Cases: 1,096 (7.1% of clustered); resolved 13.6%; DM redirect 5.3%
- Resolution types: information_provided 46%, feedback_acknowledged 9%, unresolved 8%, redirected_to_other_operator 7%, self_service 7%, clarification_requested 6%
- Confusable with: 5 journey_disruption_complaint (22%), 8 praise_positive_feedback (18%), 10 ticket_booking_query (14%)
- Auto-handle hint: `unclear` (NEEDS_REVIEW)

Representative examples:
  - `case_1205576` @VirginTrains - booked 8 seats months in advance - carriage missing - having to stand from Doncaster to London - not good enough!!!
  - `case_862994` @VirginTrains Birmingham to Glasgow 15.15 reserved seat and no carriage available @127472 #Bad-Service #NotHappy!
  - `case_908730` Seat reservation not loaded on to the seats on the @virgintrains 19.30 Euston to Glasgow Central... causing chaos!
  - `case_2437826` @VirginTrains on 15:23 from Crewe to Euston. Not enough seats.Some seats double booked.Some reserved seats showing as available.Sort it out.
  - `case_2249367` Carriage F of the 1955 Man to Eus. It's the only unreserved seating coach on the service and there are 34 reservations in it. Could you explain please @VirginTrains ? @125861

Ambiguous examples:
  - `case_2428661` @VirginTrains hi I’ll be on the delayed 17.51 train from Carlisle to London - is there unreserved seats somewhere? Thanks  _(also close to 1:service_running_enquiry)_
  - `case_2675024` @VirginTrains what’s the point of reserving seats 2 months in advance, if you end up standing on your train journey from Edinburgh to London? #fail #virgintrains #badservice #complaint  _(also close to 5:journey_disruption_complaint)_
  - `case_1649219` Worst train journey I've ever been on @VirginTrains #delayed #overcrowded #toiletsbroken + cancelled seat reservations - How do I complain? https://t.co/UnEdDFuEJN  _(also close to 5:journey_disruption_complaint)_
  - `case_1848478` @VirginTrains Idea pitch: A noisy coach. Like the opposite of a quiet coach with straw on the floor where parents with crying kids can go?  _(also close to 0:chitchat_non_support)_
  - `case_2891419` @VirginTrains Please send the conductor to coach E on the 14.40 eus to Manc. There’s some kind of mix up with reservations.  _(also close to 1:service_running_enquiry)_

Representative resolution patterns (first agent reply):
  - 10x "If you mention this to the Train Manager they can assist you further, Mike"
  - 7x "Which service are you on today exactly?"
  - 5x "If you speak to staff, they can reset it."
  - 5x "Sorry to hear about your experience, the team onboard would be best to advise further"
  - 5x "Sorry to hear of your experience onboard today. Have you managed to sit in your reserved seats?"

## Cluster 7: customer_service_complaint

Complaint about staff, customer service, unanswered complaints/phone lines, or the quality of a previous brand response.

- Cases: 1,090 (7.0% of clustered); resolved 10.9%; DM redirect 10.3%
- Resolution types: information_provided 27%, unresolved 18%, self_service 13%, redirected_to_dm 8%, feedback_acknowledged 8%, other 6%
- Confusable with: 0 chitchat_non_support (38%), 2 delay_in_progress_report (20%), 8 praise_positive_feedback (12%)
- Auto-handle hint: `likely_no` (NEEDS_REVIEW)

Representative examples:
  - `case_2120121` @VirginTrains @624733 Got a reply from East Midlands from our trip to Sheffield on 21st. Better customer service obviously.
  - `case_538290` @VirginTrains went from great service on Thurs to terrible service today - delays cannot be helped but unhelpful, rude staff can be. Wrong information aswell shame really
  - `case_1617648` Unbelievable poor customer service from virgin west coast at Euston today @virgintrains
  - `case_1317055` @VirginTrains have been speaking to different people in your customer service for DAYS and no1 has been helpful. Such disappointing service
  - `case_621624` Horrible customer service from @VirginTrains expected a lot better when you are trying to use their service but making it impossible! Poor service!

Ambiguous examples:
  - `case_1205564` Really glad I forked out 5 quid for the @VirginTrains onboard internet. If this tweet posts it will be a miracle. 😂 https://t.co/U16pfyIRUg  _(also close to 0:chitchat_non_support)_
  - `case_2419806` Good god Dial up world be faster on @VirginTrains #internet #internetservice https://t.co/5TKYzqmtCv  _(also close to 11:onboard_wifi_issue)_
  - `case_2391698` @120576 @VirginTrains @688700 And it's partly due to All The Stations as well, you won't forget them after you mentioned them in your vid about Azuma?  _(also close to 2:delay_in_progress_report)_
  - `case_1696513` rubbish first class service on the 1731 EDB - KGX tonight @VirginTrains. Short staffed so nobody to operate the microwave!!  _(also close to 4:first_class_service_issue)_
  - `case_1761431` @VirginTrains @422836 Been watching this. Amazing kudos to those just trying to get on and home. Those complaining should just step back a bit and think  _(also close to 0:chitchat_non_support)_

Representative resolution patterns (first agent reply):
  - 8x "Were you looking to make a complaint today, Victoria?"
  - 6x "It can take up to 28 days for a full response to be issued. Have you received any VT reference Fiona?"
  - 5x "Anything we can help with today?"
  - 5x "This can be escalated further with Customer Relations via the online complaints link - https://t.co/t20UbyOCSn"
  - 4x "Sorry to hear of your experience today, Ria. We'll certainly pass on your comments regarding this"

## Cluster 2: delay_in_progress_report

Customer is currently delayed or waiting and asks for news, or complains about missing announcements.

- Cases: 1,032 (6.7% of clustered); resolved 15.5%; DM redirect 1.8%
- Resolution types: information_provided 50%, compensation 10%, self_service 10%, unresolved 9%, other 6%, redirected_to_other_operator 5%
- Confusable with: 9 service_status_update_request (22%), 7 customer_service_complaint (18%), 8 praise_positive_feedback (17%)
- Auto-handle hint: `unclear` (NEEDS_REVIEW)

Representative examples:
  - `case_539584` @VirginTrains hello, any news on the 14:11 to London from Coventry? I have a ticket on this and its just saying delayed
  - `case_640910` @VirginTrains delayed coming to London this am now delayed going home:-( any departure time expected for the 2007 to Liverpool please?
  - `case_1276546` @VirginTrains 8.23 from Rugby is still at Rugby Station. We’re now over 20 minutes late. No announcement to help us. What’s happening?!
  - `case_1618183` @VirginTrains any update on the delays at Preston? Still waiting for the 13:42 to Glasgow Central. Thanks.
  - `case_767219` @VirginTrains Hello! Any news on what the delay is between Liverpool to Euston please? I’m waiting at Crewe for the 9.13. Thanks!

Ambiguous examples:
  - `case_1481049` @VirginTrains what time Is the 7-21 from peterborough to London due to arrive today with the delay?  _(also close to 9:service_status_update_request)_
  - `case_2347185` @VirginTrains what do you think about how slow it is from Stoke-on-Trent to liverpool and cost  _(also close to 9:service_status_update_request)_
  - `case_1621802` @VirginTrains When you say 'as close to your booked time as possible', would you mind clarifying what you mean? Thanks  _(also close to 10:ticket_booking_query)_
  - `case_489309` @231530 @VirginTrains Oh no. I hope you reach your destination in time. Always happens at the wrong time. Cannot work it out who you are seeing though. Must be a confirmed old dog now if I cannnot bring him/her to mind. Is it Steps?  _(also close to 8:praise_positive_feedback)_
  - `case_589106` @VirginTrains hi sorry if this has already been asked but is the 19:57 to Manchester running? If so, is it likely to be delayed? Thanks  _(also close to 9:service_status_update_request)_

Representative resolution patterns (first agent reply):
  - 6x "Delay Repay compensation is available to those who have been caught up in disruption and delayed by over 30 minutes. Please claim via our Delay Repay form here…"
  - 5x "We're awaiting an update as to when the lines will reopen, David. Live Updates can be found via https://t.co/J27w0zJVCl"
  - 4x "Which service are you on?"
  - 4x "Which service are you on this morning?"
  - 4x "Sorry to hear that. Please claim for the Delay here - https://t.co/cZj2DV3488"

## Cluster 3: delay_repay_refund_claim

Customer asks about, chases, or disputes a Delay Repay claim, refund, or compensation for a delayed or cancelled journey.

- Cases: 884 (5.7% of clustered); resolved 16.1%; DM redirect 17.6%
- Resolution types: compensation 33%, information_provided 19%, redirected_to_dm 13%, refund 11%, self_service 8%, escalated 4%
- Confusable with: 10 ticket_booking_query (35%), 7 customer_service_complaint (25%), 5 journey_disruption_complaint (16%)
- Auto-handle hint: `unclear` (NEEDS_REVIEW)

Representative examples:
  - `case_2073832` @VirginTrains :shocked at poor service, two delayed trains over 2 wks: filled in delay repay forms and was told you would get back to me. This was weeks ago, you haven’t. I want my refunds!
  - `case_503112` .@VirginTrains please can I get a refund for my 1 hr 8 min delayed train journey this evening?
  - `case_129268` @VirginTrains hi, it’s now been 28 days since my train was cancelled and I have yet to receive my money back through delay repay.
  - `case_2404550` @VirginTrains my train was delayed by 130 mins. I've done delay repay and was told it was 17 mins late so no refund. Who can sort this out for me?
  - `case_1753428` @VirginTrains Hi, any chance you can look into my delay repay refund from another cancelled train on 27th October. I've not received it yet.

Ambiguous examples:
  - `case_2188795` @VirginTrains do you accept PayPal?  _(also close to 10:ticket_booking_query)_
  - `case_496522` Manchester to London train delayed by 30mins! Ridiculous @VirginTrains I expect a refund. Won't catch my connecting train now. Well done!!  _(also close to 1:service_running_enquiry)_
  - `case_2325653` Hi @VirginTrains I haven’t received my compensation yet for a cancelled train home from Euston-Picc on 5 nov. Thought it was automatic?  _(also close to 1:service_running_enquiry)_
  - `case_1763301` @VirginTrains what is wrong with you people. Book direct with virgin. Fill in delay replay form with booking ref. Get email saying can't help as don't have ref! Twice now. Online chat says need to fill form in !  _(also close to 7:customer_service_complaint)_
  - `case_541913` @VirginTrains @244269 I've messaged and emailed about a journey weeks ago and still nothing back!  _(also close to 2:delay_in_progress_report)_

Representative resolution patterns (first agent reply):
  - 24x "It can take up to 28 days for a full response to be issued. Have you received a VT reference for this?"
  - 16x "Do you have a VT reference number you could DM us with?"
  - 9x "Delay Repay compensation is available to those who have been caught up in disruption and delayed by over 30 minutes. Please claim via our Delay Repay form here…"
  - 7x "It can take the team up to 28 days to come back to you but should be sooner than this, Zenny"
  - 7x "Please claim via our Delay Repay form here - https://t.co/M07kWkMhdd"

## Cluster 4: first_class_service_issue

First class catering, lounge, upgrade, or reduced first class service; includes some positive remarks.

- Cases: 707 (4.5% of clustered); resolved 18.1%; DM redirect 1.6%
- Resolution types: information_provided 43%, self_service 13%, feedback_acknowledged 10%, other 7%, unresolved 6%, compensation 6%
- Confusable with: 6 seat_reservation_issue (18%), 7 customer_service_complaint (16%), 0 chitchat_non_support (15%)
- Auto-handle hint: `unclear` (NEEDS_REVIEW)

Representative examples:
  - `case_496482` @VirginTrains no food or drink in first class on the 17:40 Euston to Manchester. Any chance of a partial refund or a free upgrade on a future service?
  - `case_2417615` @VirginTrains sitting in 1st class and there is no food to be offered just drinks... #firstclass #disappointed #paid for #service 😭😭😭
  - `case_2368122` Hi @VirginTrains Can I ask why there is no food or drink at all in First Class on the 15.50 from Birmingham to Euston I am on? Everybody has paid quite a lot for their tickets and we haven't even had a coffee.
  - `case_2732614` @VirginTrains on14.00 from Glasgow coach J,the food and drink and service from the staff is First Class👍
  - `case_1333028` @VirginTrains any seats in first class ‘spare’ there 8 ppl sat on the floor in coach A 12.58 train Preston to London https://t.co/6Z4lfM0y09

Ambiguous examples:
  - `case_836604` @VirginTrains no card payments on the 9:41 from Preston to Glasgow. 😱 People need coffee in the AM.  _(also close to 2:delay_in_progress_report)_
  - `case_2694762` Had to spend 10% of my monthly wage to sit on the floor of @VirginTrains for 10 hours due to overselling &amp; overcrowding.  _(also close to 7:customer_service_complaint)_
  - `case_809820` @VirginTrains @1685. No staff to help our group and wheelchair student apparently #virgintrains https://t.co/0GbmjmDgo1  _(also close to 7:customer_service_complaint)_
  - `case_1333028` @VirginTrains any seats in first class ‘spare’ there 8 ppl sat on the floor in coach A 12.58 train Preston to London https://t.co/6Z4lfM0y09  _(also close to 6:seat_reservation_issue)_
  - `case_2618608` I’m sure @VirginTrains are running a drinks Sales campaign by turning the heating up to 30c - coach F should be renamed “the oven” 😢  _(also close to 6:seat_reservation_issue)_

Representative resolution patterns (first agent reply):
  - 6x "Which service are you on today?"
  - 6x "Which service are you travelling on tomorrow, Ayden?"
  - 5x "We run a reduced First Class service on the weekends as advertised on our website"
  - 3x "We'll pass on your comments regarding this John"
  - 3x "The onboard team would be best placed to help you with this"

## Cluster 11: onboard_wifi_issue

Onboard Wi-Fi not connecting, not working after paying, or paid-for access disputes.

- Cases: 519 (3.3% of clustered); resolved 12.9%; DM redirect 2.5%
- Resolution types: information_provided 33%, troubleshooting 24%, self_service 10%, unresolved 6%, clarification_requested 6%, refund 6%
- Confusable with: 7 customer_service_complaint (18%), 5 journey_disruption_complaint (16%), 3 delay_repay_refund_claim (12%)
- Auto-handle hint: `likely_yes` (NEEDS_REVIEW)

Representative examples:
  - `case_2113438` Paid for wifi on @VirginTrains for my 2 hour trip from London and it doesn't even work when I really need it, that's cool 🙃🙄
  - `case_1421608` The WiFI on this @VirginTrains service to Birmingham is shocking, thank god I didn't pay for it.
  - `case_638720` @VirginTrains I've paid for Wi-Fi on the train to London from Glasgow but it isn't working.
  - `case_2452631` Wifi on @VirginTrains is awful and not even free
  - `case_1741866` @VirginTrains struggling to buy WiFi? Manchester- Euston train departed 17:15

Ambiguous examples:
  - `case_2937967` Oh dear @VirginTrains, what a journey. Train to London: no working plug socket, captcha not working so can’t purchase WiFi pass, card machines not working for purchasing food/drink. Not the usual standard for a £300 ticket...  _(also close to 10:ticket_booking_query)_
  - `case_443999` @220186 @VirginTrains I've given up on the hopeless app completely since the "upgrade" - now using website, Trainline or station machines. Shame - it used to be good  _(also close to 5:journey_disruption_complaint)_
  - `case_2201996` @VirginTrains Why is your free Internet and free Beam app not available when the train manager has just announced how good it is? Man 2 Lon  _(also close to 5:journey_disruption_complaint)_
  - `case_1690932` @VirginTrains “Rock Star Treatment” 19.50pm is classed as Late Evening so Dinner Service is finished. WiFi &amp; 240v socket both broken https://t.co/ZKDIWtdF4I  _(also close to 2:delay_in_progress_report)_
  - `case_1506023` @VirginTrains Are you using Opera as your web browser?!  _(also close to 7:customer_service_complaint)_

Representative resolution patterns (first agent reply):
  - 20x "What issues are you having?"
  - 11x "Have you tried disconnecting from the WiFi, reconnecting and typing https://t.co/1JEOSVPtGN into your browser's URL bar?"
  - 9x "Are you struggling to connect or with speeds, Samantha?"
  - 8x "Try entering https://t.co/1JEOSVPtGN into your browser, Jane!"
  - 8x "Please forget the network and switch off Wi-Fi. Then reconnect to “virgintrainswifi” and load https://t.co/1JEOSVPtGN"
