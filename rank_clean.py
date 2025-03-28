from openskill.models import PlackettLuce, PlackettLuceRating
from typing import TypedDict, Any, Dict, List, Tuple, Optional
from datetime import datetime
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

# Define type aliases and custom types
class PlayerInfo(TypedDict):
    player_id: str
    steam_name: str

class PlayerData(TypedDict):
    steam_name: str
    rating_data: PlackettLuceRating
    player: bool
    games_played: int
    wins: int
    history: List[Tuple[datetime, float]]  # Track (timestamp, ordinal) for each game

HistogramType = Dict[str, Dict[int, float]]

def parse_battle_report(file_path: Path) -> Tuple[Dict[str, List[PlayerInfo]], str]:
    with open(file_path) as fp:
        xml_string = fp.read()
        last_gt_index = xml_string.rfind('>')
        if last_gt_index != -1:
            xml_string = xml_string[:last_gt_index + 1]
        tree = ET.ElementTree(ET.fromstring(xml_string))
    root = tree.getroot()
    teams_data: Dict[str, List[PlayerInfo]] = {}
    
    for team_element in root.findall('./Teams/*'):
        team_id_element = team_element.find('TeamID')
        team_id = team_id_element.text if team_id_element is not None else ''
        players: List[PlayerInfo] = []
        
        for player_element in team_element.findall('./Players/*'):
            player_name_element = player_element.find('PlayerName')
            account_id_element = player_element.find('AccountId/Value')
            
            player_name = html.escape(player_name_element.text) if player_name_element is not None else ''
            player_id = html.escape(account_id_element.text) if account_id_element is not None else ''
            
            players.append({
                'player_id': player_id,
                'steam_name': player_name
            })
        
        teams_data[team_id] = players
    
    winner_element = root.find('WinningTeam')
    winner = winner_element.text if winner_element is not None else ''
    return teams_data, winner

def update_database_and_teams(
    teams_data: Dict[str, List[PlayerInfo]],
    database: Dict[str, PlayerData],
    model: PlackettLuce,
) -> Dict[str, List[PlackettLuceRating]]:
    updated_teams: Dict[str, List[PlackettLuceRating]] = {}
    
    for team_id, players in teams_data.items():
        team_players: List[PlackettLuceRating] = []
        
        for player in players:
            player_id = player['player_id']
            steam_name = player['steam_name']

            if player_id not in database:
                print(f"Found new player: {steam_name} {player_id}")
                database[player_id] = {
                    "steam_name": steam_name,
                    "rating_data": model.rating(name=player_id),
                    "player": True,
                    "games_played": 0,
                    "wins": 0,
                    "history": []  # Initialize empty history
                }
            database[player_id]['games_played'] += 1
            team_players.append(database[player_id]['rating_data'])
        
        updated_teams[team_id] = team_players
    
    return updated_teams

def process_match_result(
    winner: str,
    updated_teams: Dict[str, List[PlackettLuceRating]],
    model: PlackettLuce,
    database: Dict[str, PlayerData],
    game_time: datetime
) -> None:
    teams = updated_teams.copy()
    
    try:
        winner_team = teams.pop(winner)
        other_team = teams[next(iter(teams))]
    except KeyError:
        print("Invalid team structure for match processing")
        return

    all_participants = []
    for player_rating in winner_team + other_team:
        player_id = player_rating.name
        database[player_id]['history'].append((
            game_time,
            database[player_id]['rating_data'].ordinal()
        ))
        all_participants.append(player_id)

    rated_teams = model.rate([winner_team, other_team])
    
    for team in rated_teams:
        for player in team:
            player_id = player.name
            database[player_id]['rating_data'] = player
            database[player_id]['history'][-1] = (
                game_time,
                player.ordinal()
            )

    for player_rating in winner_team:
        player_id = player_rating.name
        database[player_id]['wins'] += 1

def update_histogram(
    histogram: HistogramType,
    database: Dict[str, PlayerData],
    game_index: int
) -> None:
    for player_id, data in database.items():
        if data['player']:
            # Changed to use player_id as key
            histogram[player_id][game_index] = data['rating_data'].ordinal()

def render_leaderboard(
    database: Dict[str, PlayerData],
    output_html: bool = True
) -> None:
    """Display sorted leaderboard and generate static HTML"""
    leaderboard = sorted(database.values(), 
                        key=lambda d: d["rating_data"].ordinal(), 
                        reverse=True)
    
    print("\nLEADERBOARD:")
    for p in leaderboard:
        print(f"{p['steam_name']:20} DLO = {p['rating_data'].ordinal():6.2f} "
              f"Matches: {p['games_played']}")

    if output_html:
        player_dir = Path('docs/player')
        player_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate individual player pages
        for player_id, data in database.items():
            render_player_page(player_id, data)        

        # Main leaderboard HTML
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
        a {{ color: #0066cc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="header">
        <img src="../dlo.webp" alt="Logo" width="150">
        <h1>Player Leaderboard</h1>
        <a href="https://openskill.me/en/stable/manual.html">ranking system info</a> 
    </div>
    
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Player</th>
                <th>DLO</th>
                <th>Matches Played</th>
            </tr>
        </thead>
        <tbody>
            {"".join(
                f'<tr><td>{i+1}</td>'
                f'<td><a href="player/{p["rating_data"].name}.html">{p["steam_name"]}</a></td>'
                f'<td>{p["rating_data"].ordinal():0.2f}</td>'
                f'<td>{p["games_played"]}</td></tr>'
                for i, p in enumerate(leaderboard)
            )}
        </tbody>
    </table>
</body>
</html>
        '''

        output_path = Path('docs/index.html')
        output_path.write_text(html_content)
        print(f"\nGenerated static site at: {output_path.absolute()}")

def render_player_page(
    player_id: str,
    player_data: PlayerData
) -> None:
    """Generate individual player page with stats and history graph"""
    img_dir = Path('docs/player/images')
    img_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate DLO history plot
    plot_path = img_dir / f'{player_id}_history.webp'
    generate_dlo_plot(player_data, plot_path)
    
    # Calculate stats
    total_games = player_data['games_played']
    losses = total_games - player_data['wins']
    win_rate = player_data['wins'] / total_games if total_games > 0 else 0
    
    html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{player_data['steam_name']} - Player Stats</title>
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
        .stats-table {{
            margin: 2rem 0;
            border-collapse: collapse;
            width: 100%;
        }}
        .stats-table td, .stats-table th {{
            padding: 0.75rem;
            border: 1px solid #ddd;
        }}
        .stats-table th {{
            background-color: #f5f5f5;
            width: 30%;
        }}
        img {{
            max-width: 800px;
            margin: 2rem 0;
        }}
        a {{ color: #0066cc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="header">
        <img src="../dlo.webp" alt="Logo" width="150">
        <h1>{player_data['steam_name']} - Player Statistics</h1>
        <a href="https://openskill.me/en/stable/manual.html">ranking system info</a> 
        | <a href="../index.html">Back to Leaderboard</a>
    </div>

    <table class="stats-table">
        <tr>
            <th>Steam Name</th>
            <td>{player_data['steam_name']}</td>
        </tr>
        <tr>
            <th>Player ID</th>
            <td>{player_id}</td>
        </tr>
        <tr>
            <th>Wins/Losses</th>
            <td>{player_data['wins']} / {losses} ({win_rate:.1%})</td>
        </tr>
        <tr>
            <th>Current DLO</th>
            <td>{player_data['rating_data'].ordinal():.2f}</td>
        </tr>
        <tr>
            <th>Mu (μ)</th>
            <td>{player_data['rating_data'].mu:.2f}</td>
        </tr>
        <tr>
            <th>Sigma (σ)</th>
            <td>{player_data['rating_data'].sigma:.2f}</td>
        </tr>
    </table>

    <h2>DLO History</h2>
    <img src="images/{player_id}_history.webp" alt="DLO history graph">
</body>
</html>
    '''

    output_path = Path(f'docs/player/{player_id}.html')
    output_path.write_text(html_content)

def generate_dlo_plot(
    player_data: PlayerData,
    output_path: Path
) -> None:
    """Generate and save DLO history plot for a player with optimized compression"""
    history = player_data['history']
    if not history:
        return

    plt.figure(figsize=(10, 5), dpi=80)  # Smaller dimensions and lower DPI
    plt.rcParams['savefig.facecolor'] = 'white'  # Remove transparent background
    
    # Unzip timestamps and values
    times, ordinals = zip(*sorted(history, key=lambda x: x[0]))
    
    plt.plot(times, ordinals, marker='o', linestyle='-', markersize=4)
    plt.title(f'DLO Rating History - {player_data["steam_name"]}')
    plt.xlabel('Date')
    plt.ylabel('DLO Rating')
    plt.grid(True, alpha=0.5)  # Lighter grid lines
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save with optimized JPEG settings (can also use WebP for better compression)
    save_kwargs = {
        'dpi': 80,
        'format': 'webp'
    }
    
    output_path = output_path.with_suffix('.webp')
    save_kwargs['format'] = 'webp'
    
    plt.savefig(output_path, **save_kwargs)
    plt.close()

def load_rank_adjustments(file_path: str = 'rank_adjustments.json') -> List[Dict[str, Any]]:
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

def apply_manual_adjustments(
    model: PlackettLuce,
    database: Dict[str, PlayerData],
    adjustments: List[Dict[str, Any]]
) -> None:
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

def main() -> None:
    model: PlackettLuce = PlackettLuce(balance=False, limit_sigma=False)
    database: Dict[str, PlayerData] = {}
    
    battle_reports = sorted(Path("/srv/BattleReports").iterdir(), key=os.path.getmtime)
    print(battle_reports)
    for index, file in enumerate(battle_reports):
        print(f"\nPROCESSING FILE: {file}")
        # Get match timestamp from file creation time
        game_time = datetime.fromtimestamp(file.stat().st_ctime)
        
        teams_data, winner = parse_battle_report(file)
        updated_teams = update_database_and_teams(teams_data, database, model)
        
        if winner not in updated_teams:
            print(f"ERROR: No valid winner in {file}")
            continue
        
        process_match_result(winner, updated_teams, model, database, game_time)

    # Apply manual adjustments
    adjustments = load_rank_adjustments()
    if adjustments:
        print("\nApplying manual adjustments:")
        apply_manual_adjustments(model, database, adjustments)
    
    render_leaderboard(database)

if __name__ == "__main__":
    main()
