# this is used for ad-hoc prompt editing

manual_prompt = \
"""
Question: what's the supreme law in the canadian law?
Candidate query A: (AND law.constitution (JOIN (R base.morelaw.legal_system.constitution) m.01qk86))
Candidate query B: (JOIN (R base.morelaw.legal_system.constitution) m.01qk86)
Which candidate matches the question intent better: A
Question: what's the supreme law in the canadian law?
Candidate query A: (AND common.thing (JOIN (R base.morelaw.legal_system.constitution) m.01qk86))
Candidate query B: (AND law.constitution (JOIN (R base.morelaw.legal_system.constitution) m.01qk86))
Which candidate matches the question intent better: B
Question: what's the supreme law in the canadian law?
Candidate query A: (COUNT (AND law.constitution (JOIN (R base.morelaw.legal_system.constitution) m.01qk86)))
Candidate query B: (AND law.constitution (JOIN (R base.morelaw.legal_system.constitution) m.01qk86))
Which candidate matches the question intent better: B
Question: what's the supreme law in the canadian law?
Candidate query A: (JOIN aviation.aircraft_model.aircraft_type m.01qk86)
Candidate query B: (JOIN (R base.morelaw.legal_system.constitution) m.01qk86)
Which candidate matches the question intent better: B
Question: the house of blues show is found in the casino owned by whom?
Candidate query A: (AND base.casinos.casino_owner (JOIN (R base.casinos.casino.owner) (JOIN base.casinos.casino.shows m.0525fd)))
Candidate query B: (JOIN (R base.casinos.casino.owner) (JOIN base.casinos.casino.shows m.0525fd))
Which candidate matches the question intent better: A
Question: the house of blues show is found in the casino owned by whom?
Candidate query A: (JOIN base.casinos.casino.shows m.0525fd)
Candidate query B: (JOIN (R base.casinos.casino.owner) (JOIN base.casinos.casino.shows m.0525fd))
Which candidate matches the question intent better: B
Question: the house of blues show is found in the casino owned by whom?
Candidate query A: (AND base.casinos.casino_owner (JOIN (R base.casinos.casino.owner) (JOIN base.casinos.casino.shows m.0525fd)))
Candidate query B: (JOIN base.casinos.casino.shows m.0525fd)
Which candidate matches the question intent better: A
Question: the house of blues show is found in the casino owned by whom?
Candidate query A: (JOIN base.conservation.protected_species_status.status_designation m.0525fd)
Candidate query B: (JOIN base.casinos.casino.shows m.0525fd)
Which candidate matches the question intent better: B
Question: the house of blues show is found in the casino owned by whom?
Candidate query A: (JOIN base.conservation.protected_species_status.status_designation m.0525fd)
Candidate query B: (JOIN base.casinos.casino.shows m.0525fd)
Which candidate matches the question intent better: B
Question: enumerate all the breeds of cat.
Candidate query A: (AND base.petbreeds.cat_breed (JOIN (R biology.domesticated_animal.breeds) m.01yrx))
Candidate query B: (AND law.constitution (JOIN (R base.morelaw.legal_system.constitution) m.01qk86))
Which candidate matches the question intent better: A
Question: what election's primaries are in 2008 republican primary, california’s 79th assembly district?
Candidate query A: (AND government.election (JOIN government.election.primaries 2008 Republican Primary, California’s 79th Assembly District)
Candidate query B: (JOIN government.political_district.elections_inv (JOIN government.political_district.elections 2008 Republican Primary, California’s 79th Assembly District)
Which candidate matches the question intent better: A
Question: which file format has the same genre as the container for html file format?
Candidate query A: (JOIN computer.file_format_genre.file_formats_inv (JOIN computer.file_format.genre_inv (JOIN computer.file_format.container_for_inv HTML File Format)))
Candidate query B: (AND computer.file_format (JOIN computer.file_format_genre.file_formats_inv (JOIN computer.file_format.genre_inv (JOIN computer.file_format.container_for_inv HTML File Format))))
Which candidate matches the question intent better: B
Question: waco cg-4a is what aircraft model?
Candidate query A: (AND aviation.aircraft_model (JOIN aviation.aircraft_model.parent_aircraft_model waco cg-4a))
Candidate query B: (AND aviation.aircraft_manufacturer (JOIN aviation.aircraft_manufacturer.aircraft_models_made waco cg-4a))
Which candidate matches the question intent better: A
Question: waco cg-4a is what aircraft model?
Candidate query A: (AND aviation.aircraft_model (JOIN aviation.aircraft_model.parent_aircraft_model waco cg-4a))
Candidate query B: (AND aviation.aircraft_manufacturer (JOIN aviation.aircraft_manufacturer.aircraft_models_made waco cg-4a))
Which candidate matches the question intent better: A
Question: waco cg-4a is what aircraft model?
Candidate query A: (AND aviation.aircraft_model (JOIN aviation.aircraft_model.parent_aircraft_model_inv (JOIN aviation.aircraft_model.parent_aircraft_model waco cg-4a)))
Candidate query B: (AND aviation.aircraft_model (JOIN aviation.aircraft_model.parent_aircraft_model waco cg-4a))
Which candidate matches the question intent better: B
Question: waco cg-4a is what aircraft model?
Candidate query A: (JOIN aviation.aircraft_model.parent_aircraft_model waco cg-4a)
Candidate query B: (AND aviation.aircraft_model (JOIN aviation.aircraft_model.parent_aircraft_model waco cg-4a))
Which candidate matches the question intent better: B
Question: what type of storage is supported by casio qv-3500ex with bayer color filter array?
Candidate query A: (AND (JOIN digicams.camera_storage_type.compatible_cameras Casio QV-3500EX) (JOIN digicams.digital_camera.supported_storage_types_inv (JOIN digicams.digital_camera.color_filter_array_type bayer)))
Candidate query B: (JOIN digicams.camera_storage_type.compatible_cameras Casio QV-3500EX)
Which candidate matches the question intent better: A
Question: what type of storage is supported by casio qv-3500ex with bayer color filter array?
Candidate query A: (JOIN digicams.digital_camera.supported_storage_types_inv (JOIN digicams.digital_camera.color_filter_array_type bayer))
Candidate query B: (AND digicams.camera_storage_type (JOIN digicams.camera_storage_type.compatible_cameras Casio QV-3500EX))
Which candidate matches the question intent better: A
Question: in the tv episode stephen bishop who performs the song?
Candidate query A: (JOIN tv.tv_series_episode.songs_inv Stephen Bishop)
Candidate query B: (JOIN tv.tv_series_episode.next_episode Stephen Bishop)
Which candidate matches the question intent better: A
Question: in the tv episode stephen bishop who performs the song?
Candidate query A: (JOIN tv.tv_series_episode.songs_inv Stephen Bishop)
Candidate query B: (JOIN tv.tv_episode_song_relationship.performers_inv (JOIN tv.tv_series_episode.songs_inv Stephen Bishop))
Which candidate matches the question intent better: B
Question: in the tv episode stephen bishop who performs the song?
Candidate query A: (JOIN tv.tv_episode_song_relationship.performers_inv (JOIN tv.tv_series_episode.songs_inv Stephen Bishop))
Candidate query B: (AND tv.tv_song_performer (JOIN tv.tv_episode_song_relationship.performers_inv (JOIN tv.tv_series_episode.songs_inv Stephen Bishop))
Which candidate matches the question intent better: B
Question: in the tv episode stephen bishop who performs the song?
Candidate query A: (AND tv.tv_song_performer (JOIN tv.tv_episode_song_relationship.performers_inv (JOIN tv.tv_series_episode.songs_inv Stephen Bishop))
Candidate query B: (AND tv.tv_song (JOIN tv.tv_episode_song_relationship.song_inv (JOIN tv.tv_series_episode.songs_inv Stephen Bishop)))
Which candidate matches the question intent better: A"""

manual_prompt_short = """
Question: what's the supreme law in the canadian law?
Candiate query A: (AND law.constitution (JOIN (R base.morelaw.legal_system.constitution) canadian law))
Candidate query B: (JOIN (R base.morelaw.legal_system.constitution) canadian law)
Which candidate matches the question intent better: A
Question: what's the supreme law in the canadian law?
Candiate query A: (AND common.thing (JOIN (R base.morelaw.legal_system.constitution) canadian law))
Candidate query B: (AND law.constitution (JOIN (R base.morelaw.legal_system.constitution) canadian law))
Which candidate matches the question intent better: B
Question: what's the supreme law in the canadian law?
Candiate query A: (JOIN aviation.aircraft_model.aircraft_type canadian law)
Candidate query B: (JOIN (R base.morelaw.legal_system.constitution) canadian law)
Which candidate matches the question intent better: B
Question: the house of blues show is found in the casino owned by whom?
Candiate query A: (AND base.casinos.casino_owner (JOIN (R base.casinos.casino.owner) (JOIN base.casinos.casino.shows the house of blues)))
Candidate query B: (JOIN (R base.casinos.casino.owner) (JOIN base.casinos.casino.shows the house of blues))
Which candidate matches the question intent better: A
Question: the house of blues show is found in the casino owned by whom?
Candiate query A: (JOIN base.casinos.casino.shows the house of blues)
Candidate query B: (JOIN (R base.casinos.casino.owner) (JOIN base.casinos.casino.shows the house of blues))
Which candidate matches the question intent better: B
Question: the house of blues show is found in the casino owned by whom?
Candiate query A: (AND base.casinos.casino_owner (JOIN (R base.casinos.casino.owner) (JOIN base.casinos.casino.shows the house of blues)))
Candidate query B: (JOIN base.casinos.casino.shows the house of blues)
Which candidate matches the question intent better: A
Question: the house of blues show is found in the casino owned by whom?
Candiate query A: (JOIN base.conservation.protected_species_status.status_designation the house of blues)
Candidate query B: (JOIN base.casinos.casino.shows the house of blues)
Which candidate matches the question intent better: B
Question: the house of blues show is found in the casino owned by whom?
Candiate query A: (JOIN base.conservation.protected_species_status.status_designation the house of blues)
Candidate query B: (JOIN base.casinos.casino.shows the house of blues)
Which candidate matches the question intent better: B
Question: enumerate all the breeds of cat.
Candidate query A: (AND base.petbreeds.cat_breed (JOIN (R biology.domesticated_animal.breeds) cat))
Candidate query B: (AND law.constitution (JOIN (R base.morelaw.legal_system.constitution) cat))
Which candidate matches the question intent better: A
Question: how many other typefaces from microsoft are designed by the same person as verdana?
Candidate query A: (COUNT (AND base.typefaces.typeface (AND (JOIN base.typefaces.typeface.typeface_creator (JOIN (R base.typefaces.typeface.typeface_creator) verdana)) (JOIN base.typefaces.typeface.foundry microsoft))))
Candidate query B: (COUNT (AND base.typefaces.typeface (JOIN (R base.typefaces.typeface_foundry.typefaces) microsoft)))
Which candidate matches the question intent better: A
Question: how many other typefaces from microsoft are designed by the same person as verdana?
Candidate query A:  (AND base.typefaces.typeface (AND (JOIN base.typefaces.typeface.typeface_creator (JOIN (R base.typefaces.typeface.typeface_creator) verdana)) (JOIN base.typefaces.typeface.foundry microsoft)))
Candidate query B: (COUNT (AND base.typefaces.typeface (AND (JOIN base.typefaces.typeface.typeface_creator (JOIN (R base.typefaces.typeface.typeface_creator) verdana)) (JOIN base.typefaces.typeface.foundry microsoft))))
Which candidate matches the question intent better: B
"""

manual_prompt_0 = """
Question: which airplanes are smartbird?
Candidate query A: (JOIN aviation.aircraft_model.manufacturer_inv SmartBird)
Candidate query B: (JOIN aviation.aircraft.model SmartBird)
Which candidate matches the question intent better: B
Question: which fictional character produced by e. nelson bridwell and is created by romeo tanghal?
Candidate query A: (JOIN fictional_universe.fictional_character_creator.fictional_characters_created_inv E. Nelson Bridwell)
Candidate query B: (JOIN film.film_story_contributor.film_story_credits_inv E. Nelson Bridwell)
Which candidate matches the question intent better: A
Question: which disney ride was designed by the same person as superman escape?
Candidate query A: (JOIN amusement_parks.ride_designer.rides Superman Escape)
Candidate query B: (JOIN amusement_parks.ride_type.rides Superman Escape)
Which candidate matches the question intent better: A
Question: what automotive designer is credited for designing the third generation ford mustang?
Candidate query A: (JOIN automotive.model.generations Third Generation Ford Mustang)
Candidate query B: (JOIN automotive.designer.automobiles_designed Third Generation Ford Mustang)
Which candidate matches the question intent better: B
Question: octopus is compatible with which dietary restriction?
Candidate query A: (JOIN food.food.nutrients_inv Octopus)
Candidate query B: (JOIN food.dietary_restriction.compatible_ingredients Octopus)
Which candidate matches the question intent better: B
Question: what is the government with an agency that has a successor of committee of european securities regulators?
Candidate query A: (JOIN government.government_agency.successor_agency_inv Committee of European Securities Regulators)
Candidate query B: (JOIN government.government_agency.predecessor_agency_inv Committee of European Securities Regulators)
Which candidate matches the question intent better: A
"""

manual_prompt_1 = """
Question: which airplanes are smartbird?
Candidate query A: (JOIN aviation.aircraft_model.aircraft_type (JOIN aviation.aircraft_type.aircraft_of_this_type SmartBird))
Candidate query B: (AND aviation.aircraft (JOIN aviation.aircraft.model SmartBird))
Which candidate matches the question intent better: B
Question: which airplanes are smartbird?
Candidate query A: (AND aviation.aircraft (JOIN aviation.aircraft.model SmartBird))
Candidate query B: (JOIN aviation.aircraft.model SmartBird)
Which candidate matches the question intent better: A
Question: which fictional character produced by e. nelson bridwell and is created by romeo tanghal?
Candidate query A: (AND (JOIN fictional_universe.fictional_character.character_created_by Romeo Tanghal) (JOIN fictional_universe.fictional_character_creator.fictional_characters_created_inv E. Nelson Bridwell))
Candidate query B: (JOIN fictional_universe.fictional_character.character_created_by Romeo Tanghal)
Which candidate matches the question intent better: A
Question: which fictional character produced by e. nelson bridwell and is created by romeo tanghal?
Candidate query A: (AND (JOIN fictional_universe.fictional_character.character_created_by Romeo Tanghal) (JOIN film.film_story_contributor.film_story_credits_inv E. Nelson Bridwell))
Candidate query B: (AND (JOIN fictional_universe.fictional_character.character_created_by Romeo Tanghal) (JOIN fictional_universe.fictional_character_creator.fictional_characters_created_inv E. Nelson Bridwell))
Which candidate matches the question intent better: B
Question: which disney ride was designed by the same person as superman escape?
Candidate query A: (JOIN amusement_parks.ride.designer (JOIN amusement_parks.ride_designer.rides Superman Escape))
Candidate query B: (AND people.person (JOIN amusement_parks.ride.designer_inv Superman Escape))
Which candidate matches the question intent better: A
Question: which disney ride was designed by the same person as superman escape?
Candidate query A: (JOIN amusement_parks.ride_designer.rides Superman Escape)
Candidate query B: (JOIN amusement_parks.ride.designer (JOIN amusement_parks.ride_designer.rides Superman Escape))
Which candidate matches the question intent better: B
Question: what automotive designer is credited for designing the third generation ford mustang?
Candidate query A: (AND automotive.designer (JOIN automotive.designer.automobiles_designed Third Generation Ford Mustang))
Candidate query B: (JOIN automotive.generation.designer (JOIN automotive.generation.designer_inv Third Generation Ford Mustang))
Which candidate matches the question intent better: A
Question: octopus is compatible with which dietary restriction?
Candidate query A: (AND food.dietary_restriction (JOIN food.dietary_restriction.compatible_ingredients Octopus))
Candidate query B: (JOIN dining.cuisine.restaurant_inv (JOIN food.ingredient.incompatible_with_dietary_restrictions_inv Octopus))
Which candidate matches the question intent better: A
Question: octopus is compatible with which dietary restriction?
Candidate query A: (JOIN food.dietary_restriction.compatible_ingredients Octopus)
Candidate query B: (AND food.dietary_restriction (JOIN food.dietary_restriction.compatible_ingredients Octopus))
Which candidate matches the question intent better: B
Question: what automotive designer is credited for designing the third generation ford mustang?
Candidate query A: (JOIN automotive.designer.automobiles_designed Third Generation Ford Mustang)
Candidate query B: (AND automotive.designer (JOIN automotive.designer.automobiles_designed Third Generation Ford Mustang))
Which candidate matches the question intent better: B
Question: what is the government with an agency that has a successor of committee of european securities regulators?
Candidate query A: (JOIN government.governmental_body.body_this_is_a_component_of_inv (JOIN government.government_agency.successor_agency_inv Committee of European Securities Regulators))
Candidate query B: (JOIN government.government_agency.government_inv (JOIN government.government_agency.successor_agency_inv Committee of European Securities Regulators))
Which candidate matches the question intent better: B
Question: what is the government with an agency that has a successor of committee of european securities regulators?
Candidate query A: (JOIN government.government_agency.government_inv (JOIN government.government_agency.successor_agency_inv Committee of European Securities Regulators))
Candidate query B: (JOIN government.government_agency.successor_agency_inv Committee of European Securities Regulators)
Which candidate matches the question intent better: A
"""

manual_prompt_2 = """
Question: which airplanes are smartbird?
Candidate query A: (AND aviation.aircraft (JOIN aviation.aircraft.model SmartBird))
Candidate query B: (AND aviation.aircraft_model (JOIN aviation.aircraft_model.aircraft_type (JOIN aviation.aircraft_type.aircraft_of_this_type SmartBird)))
Which candidate matches the question intent better: A
Question: which fictional character produced by e. nelson bridwell and is created by romeo tanghal?
Candidate query A: (AND fictional_universe.fictional_character (AND (JOIN fictional_universe.fictional_character.character_created_by Romeo Tanghal) (JOIN fictional_universe.fictional_character_creator.fictional_characters_created_inv E. Nelson Bridwell)))
Candidate query B: (AND (JOIN fictional_universe.fictional_character.character_created_by Romeo Tanghal) (JOIN fictional_universe.fictional_character_creator.fictional_characters_created_inv E. Nelson Bridwell))
Which candidate matches the question intent better: A
Question: which fictional character produced by e. nelson bridwell and is created by romeo tanghal?
Candidate query A: (JOIN fictional_universe.fictional_character.character_created_by_inv (AND (JOIN fictional_universe.fictional_character.character_created_by Romeo Tanghal) (JOIN fictional_universe.fictional_character_creator.fictional_characters_created_inv E. Nelson Bridwell)))
Candidate query B: (AND fictional_universe.fictional_character (AND (JOIN fictional_universe.fictional_character.character_created_by Romeo Tanghal) (JOIN fictional_universe.fictional_character_creator.fictional_characters_created_inv E. Nelson Bridwell)))
Which candidate matches the question intent better: B
Question: which disney ride was designed by the same person as superman escape?
Candidate query A: (AND amusement_parks.disney_ride (JOIN amusement_parks.ride.designer (JOIN amusement_parks.ride_designer.rides Superman Escape)))
Candidate query B: (JOIN amusement_parks.ride.designer (JOIN amusement_parks.ride_designer.rides Superman Escape))
Which candidate matches the question intent better: A
Question: which disney ride was designed by the same person as superman escape?
Candidate query A: (JOIN amusement_parks.park.rides (JOIN amusement_parks.ride.designer (JOIN amusement_parks.ride_designer.rides Superman Escape))))
Candidate query B: (AND amusement_parks.disney_ride (JOIN amusement_parks.ride.designer (JOIN amusement_parks.ride_designer.rides Superman Escape)))
Which candidate matches the question intent better: B
Question: what automotive designer is credited for designing the third generation ford mustang?
Candidate query A: (AND automotive.generation (JOIN automotive.generation.designer (JOIN automotive.designer.automobiles_designed Third Generation Ford Mustang)))
Candidate query B: (AND automotive.designer (JOIN automotive.designer.automobiles_designed Third Generation Ford Mustang))
Which candidate matches the question intent better: B
Question: octopus is compatible with which dietary restriction?
Candidate query A: (AND food.ingredient (JOIN food.dietary_restriction.incompatible_ingredients_inv (JOIN food.ingredient.compatible_with_dietary_restrictions_inv Octopus)))
Candidate query B: (AND food.dietary_restriction (JOIN food.dietary_restriction.compatible_ingredients Octopus))
Which candidate matches the question intent better: B
Question: what is the government with an agency that has a successor of committee of european securities regulators?
Candidate query A: (AND government.government (JOIN government.government_agency.government_inv (JOIN government.government_agency.successor_agency_inv Committee of European Securities Regulators)))
Candidate query B: (AND government.governmental_body (JOIN government.government_agency.government_inv (JOIN government.government_agency.successor_agency_inv Committee of European Securities Regulators)))
Which candidate matches the question intent better: A
Question: what is the government with an agency that has a successor of committee of european securities regulators?
Candidate query A: (AND engineering.piston_configuration (JOIN engineering.piston_engine.piston_configuration_inv (JOIN engineering.engine.variation_of_inv Rolls-Royce Merlin I)))
Candidate query B: (JOIN engineering.piston_engine.piston_configuration_inv (JOIN engineering.engine.variation_of_inv Rolls-Royce Merlin I))
Which candidate matches the question intent better: A
Question: what is the government with an agency that has a successor of committee of european securities regulators?
Candidate query A: (JOIN government.government_agency.government_inv (JOIN government.government_agency.successor_agency_inv Committee of European Securities Regulators))
Candidate query B: (AND government.government (JOIN government.government_agency.government_inv (JOIN government.government_agency.successor_agency_inv Committee of European Securities Regulators)))
Which candidate matches the question intent better: B
"""
