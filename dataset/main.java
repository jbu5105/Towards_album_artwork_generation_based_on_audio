package fm.last.musicbrainz.coverart.impl;

import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.function.Consumer;

import org.apache.commons.io.FileUtils;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.junit.Test;

import fm.last.musicbrainz.coverart.CoverArt;
import fm.last.musicbrainz.coverart.CoverArtArchiveClient;
import fm.last.musicbrainz.coverart.CoverArtImage;


public class main {

int stepsCounter_pop = 0;
int stepsCounter_rock = 0;
int stepsCounter_metal = 0;
int stepsCounter_rb = 0;
int stepsCounter_newage = 0;
int stepsCounter_blues = 0;
int stepsCounter_rap = 0;
int stepsCounter_elec = 0;
int stepsCounter_country = 0;
int stepsCounter_punk = 0;
int stepsCounter_folk = 0;
int stepsCounter_reggae = 0;
int stepsCounter_jazz = 0;
int stepsCounter_indie = 0;
int stepsCounter_world = 0;
boolean download = false;
String folder = "";


	public void fetchFiles(File dir) {
		if (dir.isDirectory()) {
			for (File file : dir.listFiles()) {
				fetchFiles(file);
			}
		}else {
			String file_path = dir.getAbsolutePath();
			String file_name = dir.getName();
			String song_id = file_name.split("\\.")[0];
			String repeat = song_id.split("-")[5];

			if (repeat.equals("0")) {
	            JSONParser parser = new JSONParser();
	    		try {
	    			Object obj = parser.parse(new FileReader(file_path));
	     
	    			// A JSON object. Key value pairs are unordered. JSONObject supports java.util.Map interface.
	    			JSONObject jsonObject = (JSONObject) obj;
	     
	    			// A JSON array. JSONObject supports java.util.List interface.
	    			JSONObject jsonObjectmeta = (JSONObject) jsonObject.get("metadata");
	    			JSONObject jsonObjecttags = (JSONObject) jsonObjectmeta.get("tags");
	    			@SuppressWarnings("unchecked")
	    			List<String> genrelist = (List<String>) jsonObjecttags.get("genre");
					@SuppressWarnings("unchecked")
					List<String> idlist = (List<String>) jsonObjecttags.get("musicbrainz_albumid");
					@SuppressWarnings("unchecked")
					List<String> albumlist = (List<String>) jsonObjecttags.get("album");
					@SuppressWarnings("unchecked")
					List<String> artistlist = (List<String>) jsonObjecttags.get("artist");
	    			
					String id = idlist.get(0);
        			String genre = genrelist.get(0);
        			String album = albumlist.get(0);
        			String artist = artistlist.get(0);
        			
        			//System.out.println(genre);
        			
        			if ((genre != null)	&& (id != null) && (album != null) && (artist != null)){	
        					
						CoverArtArchiveClient client = new DefaultCoverArtArchiveClient();
	        			UUID mbid = UUID.fromString(id);
	
	        			CoverArt coverArt = null;
	        			try {
	        			  coverArt = client.getByMbid(mbid);
	        			  if (coverArt != null) {
	        				
	        				  if (genre.equals("Rap") || genre.equals("Hip-Hop") || genre.equals("Rap & Hip-Hop")) {
	              				
	  	        				stepsCounter_rap=stepsCounter_rap+1;
	  	        				download = true;
	  	        				folder = "rap";
	  	        				System.out.println("COUNT " + genre + " = " + stepsCounter_rap + " | " + "File: " + song_id);
	          				
	          				}else if(genre.equals("Rock") || genre.equals("rock")) {
	          				
	  	        				stepsCounter_rock=stepsCounter_rock+1;
	  	        				download = true;
	  	        				folder = "rock";
	  	        				System.out.println("COUNT " + genre + " = " + stepsCounter_rock + " | " + "File: " + song_id);
	          				
	          				}else if(genre.equals("r&b") || genre.equals("R&B")) {
	          				
	  	        				stepsCounter_rb=stepsCounter_rb+1;
	  	        				download = true;
	  	        				folder = "r&b";
	  	        				System.out.println("COUNT " + genre + " = " + stepsCounter_rb + " | " + "File: " + song_id);
	          				
	          				}else if(genre.equals("Blues") || genre.equals("blues")) {
	          				
	  	        				stepsCounter_blues=stepsCounter_blues+1;
	  	        				download = true;
	  	        				folder = "blues";
	  	        				System.out.println("COUNT " + genre + " = " + stepsCounter_blues + " | " + "File: " + song_id);
	          				
	          				}else if(genre.equals("New Age") || genre.equals("new age")) {
	          				
	  	        				stepsCounter_newage=stepsCounter_newage+1;
	  	        				download = true;
	  	        				folder = "new_age";
	  	        				System.out.println("COUNT " + genre + " = " + stepsCounter_newage + " | " + "File: " + song_id);
	          				
	          				}else if(genre.equals("Pop") || genre.equals("pop")) {
	  	        				stepsCounter_pop=stepsCounter_pop+1;
	  	        				download = true;
	  	        				folder = "pop";
	  	        				System.out.println("COUNT " + genre + " = " + stepsCounter_pop + " | " + "File: " + song_id);
	          				
	          				}else if(genre.equals("Electronic") || genre.equals("Electronica") || genre.equals("electronic")) {
	          				
	  	        				stepsCounter_elec=stepsCounter_elec+1;
	  	        				download = true;
	  	        				folder = "electronic";
	  	        				System.out.println("COUNT " + genre + " = " + stepsCounter_elec + " | " + "File: " + song_id);
	          				
	          				}else if(genre.contains("Metal") || genre.contains("metal")) {
	          					
	  	        				stepsCounter_metal=stepsCounter_metal+1;
	  	        				download = true;
	  	        				folder = "metal";
	  	        				System.out.println("COUNT " + genre + " = " + stepsCounter_metal + " | " + "File: " + song_id);
	          				
	          				}else if(genre.equals("Country")) {
	          					
	  	        				stepsCounter_country=stepsCounter_country+1;
	  	        				download = true;
	  	        				folder = "country";
	  	        				System.out.println("COUNT " + genre + " = " + stepsCounter_country + " | " + "File: " + song_id);
	          				}else if(genre.equals("Punk") || genre.equals("punk")) {
	          					
	  	        				stepsCounter_punk=stepsCounter_punk+1;
	  	        				download = true;
	  	        				folder = "punk";
	  	        				System.out.println("COUNT " + genre + " = " + stepsCounter_punk + " | " + "File: " + song_id);
	          				}else if(genre.equals("Folk") || genre.equals("folk")) {
	          					
	  	        				stepsCounter_folk=stepsCounter_folk+1;
	  	        				download = true;
	  	        				folder = "folk";
	  	        				System.out.println("COUNT " + genre + " = " + stepsCounter_folk + " | " + "File: " + song_id);
	          				}else if(genre.equals("Reggae") || genre.equals("reggae")) {
	          					
	  	        				stepsCounter_reggae=stepsCounter_reggae+1;
	  	        				download = true;
	  	        				folder = "reggae";
	  	        				System.out.println("COUNT " + genre + " = " + stepsCounter_reggae + " | " + "File: " + song_id);
	          				}else if(genre.equals("Jazz") || genre.equals("jazz")) {
	          					
	  	        				stepsCounter_jazz=stepsCounter_jazz+1;
	  	        				download = true;
	  	        				folder = "jazz";
	  	        				System.out.println("COUNT " + genre + " = " + stepsCounter_jazz + " | " + "File: " + song_id);
	          				
	          				}else if(genre.equals("Indie") || genre.equals("Indie Rock") || genre.equals("Indie Pop")) {
	          					
	  	        				stepsCounter_indie=stepsCounter_indie+1;
	  	        				download = true;
	  	        				folder = "indie";
	  	        				System.out.println("COUNT " + genre + " = " + stepsCounter_indie + " | " + "File: " + song_id);
	          				}else if(genre.equals("World") || genre.equals("world")) {
	          					
	  	        				stepsCounter_world=stepsCounter_world+1;
	  	        				download = true;
	  	        				folder = "world";
	  	        				System.out.println("COUNT " + genre + " = " + stepsCounter_world + " | " + "File: " + song_id);
	          				}else {
	          					download = false;
	          				}
	        				  
	        				if (download) {
		        				
		        			    for (CoverArtImage coverArtImage : coverArt.getImages()) {
		        			    	if(coverArtImage.isFront()) {
		        			    	  //Copy audio file
		        			    	  Path temp = Files.copy (Paths.get(file_path), Paths.get("E:/audio/" + folder + "/" + file_name)); 				    			      
				        			  if(temp != null) System.out.println("File moved successfully"); 
				    				  else System.out.println("WARNING | Failed to  the file");
				        			  //Get cover file
		        				      File output = new File("E:/covers/"+ folder + "/" + song_id + "__" + id + ".jpg");
		        				      FileUtils.copyInputStreamToFile(coverArtImage.getImage(), output);
		        			    	}
		        			    }
	        				}	  
	        			}
	        			} catch (Exception e) {
	        			  // ...
	        			}
			        		
		        			download = false;
        				}

	    		}catch (Exception e) {
	    			//e.printStackTrace();
	    		}
			}
		}
	}

	@Test
	public void test() {
		File file = new File("acousticbrainz-lowlevel-json-20150129/lowlevel/");
		fetchFiles(file);
	}
		
}
