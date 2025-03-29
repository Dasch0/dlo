from openskill.models import PlackettLuce
import os
import xml.etree.ElementTree as ET
import glob
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import random
import string
from pathlib import Path
import json
import html

def parse_battle_report(file_path):
    with open(file_path) as fp:
        xml_string = fp.read()
        last_gt_index = xml_string.rfind('>')
        if last_gt_index != -1:
            xml_string = xml_string[:last_gt_index + 1]
        tree = ET.ElementTree(ET.fromstring(xml_string))
    root = tree.getroot()
    teams_data = {}
    
    for team_element in root.findall('./Teams/*'):
        team_id = team_element.find('TeamID').text
        players = []
        
        for player_element in team_element.findall('./Players/*'):
            player_name = html.escape(player_element.find('PlayerName').text)
            player_id = html.escape(player_element.find('AccountId').find('Value').text)
            players.append({'player_id': player_id, 'steam_name': player_name})
        
        teams_data[team_id] = players
    
    winner = root.find('WinningTeam').text
    return teams_data, winner

def update_database_and_teams(teams_data, database, model, histogram):
    updated_teams = {}
    
    for team_id, players in teams_data.items():
        team_players = []
        team_ids = []
        
        # Process real players
        for player in players:
            player_id = player['player_id']
            steam_name = player['steam_name']

            if player_id not in database:
                print(f"Found new player: {steam_name} {player_id}")
                database[player_id] = {
                    "steam_name": steam_name,
                    "rating_data": model.rating(name=player_id),
                    "player": True,
                    "games_played": 0
                }
                if steam_name not in histogram:
                    histogram[steam_name] = {}
            database[player_id]['games_played'] = database[player_id]['games_played'] + 1
            team_players.append(database[player_id]['rating_data'])
            team_ids.append(player_id)
        
        updated_teams[team_id] = team_players
    
    return updated_teams

def process_match_result(winner, updated_teams, model, database):
    teams = updated_teams.copy()
    
    try:
        winner_team = teams.pop(winner)
        other_team = teams[next(iter(teams))]
    except KeyError:
        print("Invalid team structure for match processing")
        return

    rated_teams = model.rate([winner_team, other_team])
    
    for team in rated_teams:
        for player in team:
            database[player.name]['rating_data'] = player

def update_histogram(histogram, database, game_index):
    for player_id, data in database.items():
        if data['player']:
            steam_name = data['steam_name']
            histogram[steam_name][game_index] = data['rating_data'].ordinal()

# Modified render function
def render_leaderboard(database, output_html=True):
    """Display sorted leaderboard and generate static HTML"""
    leaderboard = sorted(database.values(), 
                        key=lambda d: d["rating_data"].ordinal(), 
                        reverse=True)
    
    print("\nLEADERBOARD:")
    for p in leaderboard:
        print(f"{p['steam_name']:20} DLO = {p['rating_data'].ordinal():6.2f} "
                f"games_played = {p['games_played']} "
              f"mu = {p['rating_data'].mu:6.2f} "
              f"sigma = {p['rating_data'].sigma:6.2f}")

    if output_html:
        html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DLO Leaderboard</title>
    <style>
        body {{ 
            font-family: monospace;
            margin: 2rem;
            background-color: white;
            color: black;
        }}
        .header {{
            border-bottom: 2px solid black;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 0.5rem;
            border: 1px solid #ddd;
            text-align: left;
        }}
        th {{
            background-color: #f5f5f5;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <div class="header">
        <img src="dlo.webp" alt="Logo" width="150">
        <h1>Player Leaderboard</h1>
        <a href="https://openskill.me/en/stable/manual.html">ranking system info</a> 
    </div>
    
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Player</th>
                <th>DLO</th>
                <th>Games Played</th>
                <th>Mu (μ)</th>
                <th>Sigma (σ)</th>
            </tr>
        </thead>
        <tbody>
            {"".join(
                f'<tr><td>{i+1}</td><td>{p["steam_name"]}</td><td>{p["rating_data"].ordinal():0.2f}</td><td>{p["games_played"]}</td><td>{p["rating_data"].mu:0.2f}</td><td>{p["rating_data"].sigma:0.2f}</td></tr>'
                for i, p in enumerate(leaderboard)
            )}
        </tbody>
    </table>
</body>
</html>
        '''

        output_path = Path('docs/index.html')
        output_path.parent.mkdir(exist_ok=True, parents=True)
        output_path.write_text(html_content)
        print(f"\nGenerated static site at: {output_path.absolute()}")

def plot_histogram(histogram):
    df = pd.DataFrame(histogram)
    plt.figure(figsize=(12, 6))
    df.plot(kind='line')
    plt.title('Ordinals over games played Balance = False')
    plt.xlabel('Games')
    plt.ylabel('Ordinal')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.show()

def generate_random_players(num_players):
    players = []
    for _ in range(num_players):
        player_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        steam_name = ''.join(random.choices(string.ascii_uppercase, k=8))
        players.append({'player_id': player_id, 'steam_name': steam_name})
    return players

def create_special_players(num = 4):
    return [
        {'player_id': f'special_{i}', 'steam_name': f'Special_{i}'}
        for i in range(num)
    ]

def generate_simulated_game(special_players, random_players, stack_rate=0.8, special_win_prob=.95):
    random_player_selection = random.sample(random_players, 8)
    special_player_selection = random.sample(special_players, 8)
    if random.random() < stack_rate:
        special_team = {
            'team_id': 'TeamA',
            'players': special_player_selection[0:4]
        }
        teamA_giga = True
    else:
        special_team = {
            'team_id': 'TeamA',
            'players': random_player_selection[0:4]
        }
        teamA_giga = False
        
    if random.random() < stack_rate:
        opponent_team = {
            'team_id': 'TeamB',
            'players': random_player_selection[4:8] 
        }
        teamB_giga = False
    else:
        opponent_team = {
            'team_id': 'TeamB',
            'players': special_player_selection[4:8] 
        }
        teamB_giga = True

    print("DEBUG_______")
    print(len(special_team['players']))
    print(len(opponent_team['players']))

    if teamA_giga and teamB_giga:
        winrate = 0.5
    elif teamA_giga and not teamB_giga:
        winrate = 0.8
    elif not teamA_giga and teamB_giga:
        winrate = 0.2
    else:
        winrate = 0.5
    
    winner = special_team['team_id'] if random.random() < winrate else opponent_team['team_id']
    
    teams_data = {
        special_team['team_id']: special_team['players'],
        opponent_team['team_id']: opponent_team['players']
    }
    
    return teams_data, winner

def simulate_games(database, model, histogram, num_games=500):
    special_players = create_special_players(16)
    random_players = generate_random_players(50)
    
    for p in special_players:
        if p['player_id'] not in database:
            database[p['player_id']] = {
                "steam_name": p['steam_name'],
                "rating_data": model.rating(name=p['player_id']),
                "player": True
            }
            histogram[p['steam_name']] = {}
            
    
    for game_idx in range(num_games):
        teams_data, winner = generate_simulated_game(special_players, random_players)
        updated_teams = update_database_and_teams(teams_data, database, model, histogram)
        
        if winner not in updated_teams:
            continue  # Skip invalid games
            
        process_match_result(winner, updated_teams, model, database)
        update_histogram(histogram, database, game_idx)

def load_rank_adjustments(file_path='rank_adjustments.json'):
    """Load manual rating adjustments from JSON file"""
    try:
        with open(file_path, 'r') as f:
            adjustments = json.load(f)
        return adjustments
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        return []

def apply_manual_adjustments(model, database, adjustments):
    """Apply manual rating adjustments to players"""
    for adj in adjustments:
        steam_id = adj['steam_id']
        if steam_id in database:
            # Preserve original rating
            original = database[steam_id]['rating_data']
            
            # Create new rating with adjustment
            adjusted_rating = model.rating(
                mu=original.mu + adj['mu_adjustment'],
                sigma=original.sigma
            )
            
            # Update database
            database[steam_id]['rating_data'] = adjusted_rating
            print(f"Adjusted {adj['steam_name']} ({steam_id}): "
                  f"μ {original.mu:.2f} → {adjusted_rating.mu:.2f} "
                  f"({adj['mu_adjustment']:+.2f}) - {adj['reason_for_adjustment']}")
        else:
            print(f"Warning: Player {adj['steam_name']} ({steam_id}) not found") 

def main():
    model = PlackettLuce(balance=False, limit_sigma=False)
    database = {}
    histogram = {}
    
    battle_reports = sorted(Path("/srv/BattleReports").iterdir(), key=os.path.getmtime)
    print(battle_reports)
    for index, file in enumerate(battle_reports):
        print(f"\nPROCESSING FILE: {file}")
        teams_data, winner = parse_battle_report(file)
        updated_teams = update_database_and_teams(teams_data, database, model, histogram)
        
        if winner not in updated_teams:
            print(f"ERROR: No valid winner in {file}")
            continue
        
        process_match_result(winner, updated_teams, model, database)
        update_histogram(histogram, database, index)

    # Apply manual adjustments
    adjustments = load_rank_adjustments()
    if adjustments:
        print("\nApplying manual adjustments:")
        apply_manual_adjustments(model, database, adjustments)
    
    render_leaderboard(database)

if __name__ == "__main__":
    main()
