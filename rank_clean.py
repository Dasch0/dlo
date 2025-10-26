from jinja2 import Environment, FileSystemLoader, select_autoescape

from openskill.models import PlackettLuce, PlackettLuceRating
from typing import TypedDict, Any, Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import shutil
import os
import sys
import time
import xml.etree.ElementTree as ET
import glob
import itertools
import random
import string
from pathlib import Path
import json
import html
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nfcli import determine_output_png, init_logger, load_path, nfc_theme
from nfcli.parsers import parse_any, parse_mods
import math
import pickle

# jinja setup
def datetime_format(value, fmt="%Y-%m-%d %H:%M:%S"):
    return value.strftime(fmt)

def float_round(value, precision=2):
    return round(value, precision)

def float_to_percent(value, precision=2):
    return f"{value * 100:.1f}%"

def get_template_depth(output_path: Path) -> int:
    """Calculate depth relative to docs directory"""
    try:
        rel_path = output_path.relative_to(Path("docs"))
        return len(rel_path.parent.parts)
    except ValueError:
        return 0

jinja_env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(["html", "xml"])
)

jinja_env.filters["datetime_format"] = datetime_format
jinja_env.filters["float_round"] = float_round
jinja_env.filters["float_to_percent"] = float_to_percent

class FleetEntry(TypedDict):
    time: datetime
    fleet_file: Path
    fleet_image: Path

class PlayerInfo(TypedDict):
    player_id: str
    steam_name: str
    faction: str
    fleet: FleetEntry

class PlayerData(TypedDict):
    steam_name: str
    rating_data: PlackettLuceRating
    score: float
    player: bool
    games_played: int
    wins: int
    ans_games: int
    ans_wins: int
    osp_games: int
    osp_wins: int
    history: List[Tuple[datetime, float]]
    teammates: Dict[str, Dict[str, int]]

class MatchData(TypedDict):
    valid: bool
    time: datetime
    teams: Dict[str, List[PlayerInfo]]
    winning_team: str
    avg_dlo: float
    match_quality: float

HistogramType = Dict[str, Dict[int, float]]

def collect_fleet_files():
    """store fleet files from server"""
    fleet_file_dir = Path("/srv/steam/.steam/steam/steamapps/common/NEBULOUS Dedicated Server/Saves/Fleets/")

    for fleet_file_path in fleet_file_dir.iterdir():
        if not fleet_file_path.name.endswith('.fleet'):
            continue
        print("INFO: saving fleet file:", fleet_file_path.name)
        shutil.copy(fleet_file_path, (Path(__file__).parent).joinpath('docs/fleets'))
        # rename the copied file so it has a proper timestamp to associate with skirmish reports later
        original_creation_time = os.path.getctime(fleet_file_path)
        new_name = f"{fleet_file_path.name.split('_')[0]}_{original_creation_time}.fleet"
        dirpath = (Path(__file__).parent).joinpath('docs/fleets')
        os.rename(os.path.join(dirpath, fleet_file_path.name), os.path.join(dirpath, new_name))

def generate_fleet_images():
    fleet_file_dir = (Path(__file__).parent).joinpath('docs/fleets')
    for fleet_file_path in fleet_file_dir.iterdir():
        if not fleet_file_path.name.endswith('.fleet'):
            continue
        output_file = determine_output_png(str(fleet_file_path))
        if os.path.exists(os.path.join(fleet_file_dir, output_file)):
            continue
        print("INFO: generating fleet fleet image for", fleet_file_path)
        xml_data = load_path(str(fleet_file_path))
        entity = parse_any(str(fleet_file_path), xml_data)
        if entity:
            entity.write(os.path.join(fleet_file_dir, output_file))

def parse_battle_report(file_path: Path, fleet_lut: Dict[str, List[FleetEntry]]) -> MatchData:
    ANS_HULLKEYS = [
        'Stock/Sprinter Corvette',
        'Stock/Raines Frigate',
        'Stock/Keystone Destroyer',
        'Stock/Vauxhall Light Cruiser',
        'Stock/Axford Heavy Cruiser',
        'Stock/Solomon Battleship',
        'Stock/Levy Escort Carrier']

    OSP_HULLKEYS = [
        'Stock/Shuttle',
        'Stock/Tugboat',
        'Stock/Journeyman',
        'Stock/Bulk Feeder',
        'Stock/Ore Carrier',
        'Stock/Ocello Cruiser',
        'Stock/Bulk Hauler',
        'Stock/Container Hauler',
        'Stock/Container Hauler Refit']

    FORTUNA_HULLKEYS = [
        'Escalation - Lachesis Atoll/MW_Lyvitan',
        'Escalation - Lachesis Atoll/CR_Wulver',
        'Escalation - Lachesis Atoll/SR_Hylander',
        'Escalation - Lachesis Atoll/SL_Akkogan',
        'Escalation - Lachesis Atoll/PGB_Skelligan',
    ]

    game_time = parse_skirmish_report_datetime(Path(os.path.split(file_path)[1]))

    test_time_str = "2025-04-26 05:58:07"
    date_format = "%Y-%m-%d %H:%M:%S"
    test_time = datetime.strptime(test_time_str, date_format)

    with open(file_path) as fp:
        xml_string = fp.read()
        # handle edge cases where xml encoding format doesn't match or extra data is leftover after BR
        last_gt_index = xml_string.rfind('>')
        if last_gt_index != -1:
            xml_string = xml_string[:last_gt_index + 1]
        tree = ET.ElementTree(ET.fromstring(xml_string))
    root = tree.getroot()

    # validate basic params about game. if start timestamp is 0 game didn't start. If game duration is super long or super short something probably went wrong
    if int(root.find('GameStartTimestamp').text) == 0 or int(root.find('GameDuration').text) > 7000 or int(root.find('GameDuration').text) < 200 or not bool(root.find('GameFinished').text):
        return {
            "valid": False,
            "time": game_time,
            "teams": {},
            "winning_team": 'None',
            "avg_dlo": 0.0,
            "match_quality": 0.0}

    teams_data: Dict[str, List[PlayerInfo]] = {}
    
    for team_element in root.findall('./Teams/*'):
        team_id_element = team_element.find('TeamID')
        team_id = team_id_element.text if team_id_element is not None else ''
        players: List[PlayerInfo] = []

        team_faction = ''

        for player_element in team_element.findall('./Players/*'):
            player_name_element = player_element.find('PlayerName')
            account_id_element = player_element.find('AccountId/Value')

            player_name = html.escape(player_name_element.text) if player_name_element is not None else ''
            player_id = html.escape(account_id_element.text) if account_id_element is not None else ''

            # find out what team the player was on. This then updates the team_faction which gets applied after all players are parsed
            # This ensures even if one player DCs and loses all ships, we still report their original faction correctly
            player_hullkeys = player_element.findall('./Ships/ShipBattleReport/HullKey')
            is_player_ANS = all(k.text in ANS_HULLKEYS for k in player_hullkeys) and len(player_hullkeys)
            is_player_OSP = all(k.text in OSP_HULLKEYS for k in player_hullkeys) and len(player_hullkeys)
            is_player_FORTUNA = all(k.text in FORTUNA_HULLKEYS for k in player_hullkeys) and len(player_hullkeys)

            if is_player_ANS and team_faction not in ['OSP', 'FORTUNA']:
                team_faction = 'ANS'
            if is_player_OSP and team_faction not in ['ANS', 'FORTUNA']:
                team_faction = 'OSP'
            if is_player_FORTUNA and team_faction not in ['ANS', 'OSP']:
                team_faction = 'FORTUNA'

            player = {
                'player_id': player_id,
                'steam_name': player_name,
            }
            player = find_player_fleet(game_time, player, fleet_lut)
            players.append(player)

        # assign team faction at the end to catch any players who had zero ships in the battle report
        for p in players:
            p['faction'] = team_faction
       
        for p in players:
            assert p['faction'] in ['ANS', 'OSP', 'FORTUNA']

        teams_data[team_id] = players

    winner_element = root.find('WinningTeam')
    winner = winner_element.text if winner_element is not None else ''
    return {
            "valid": True,
            "time": game_time,
            "teams": teams_data,
            "winning_team": winner,
            "avg_dlo": 0.0,
            "match_quality": 0.0
            }

def build_fleet_lut() -> Dict[str, List[FleetEntry]]:
    fleet_dir_path = (Path(__file__).parent).joinpath('docs/fleets')
    fleets = os.listdir(fleet_dir_path)

    fleet_lut = {}

    for fleet in fleets:
        if not fleet.endswith('.fleet'):
            continue
        tokens = fleet.split('_')
        steam_id = tokens[0]
        timestamp = datetime.fromtimestamp(float(tokens[-1].strip('.fleet')))
        png_path = determine_output_png(fleet)
        if steam_id not in fleet_lut:
            fleet_lut[steam_id] = []
        else:
            fleet_lut[steam_id].append({"time": timestamp, "fleet_file": fleet, "fleet_image": png_path})

    return fleet_lut

def find_player_fleet(game_time: datetime, player: PlayerInfo, fleet_lut: Dict[str, List[FleetEntry]]) -> PlayerInfo:
    ''' find associated player fleets and add them to match data'''

    fleet_dir_path = (Path(__file__).parent).joinpath('docs/fleets')
    fleets = os.listdir(fleet_dir_path)

    candidate_fleets = fleet_lut.get(player["player_id"], [])

    best_diff = None
    best_idx = None
    for idx, fleet in enumerate(candidate_fleets): 
        # find the fleet with the closest timestamp to right before the game
        diff = fleet['time'] - game_time 
        if timedelta(minutes=-120) <= diff <= timedelta(minutes=0):
            if best_diff is None:
                best_diff = diff
                best_idx = idx
            if best_diff < diff: # since all diffs are negative, ideal diff is the largest one
                best_dff = diff
                best_idx = idx
    
    if best_idx is None:
        print('INFO: no valid fleet found for', player['steam_name'], 'at', game_time)
        return player
    else:
        # pop the fleet off the list so we don't end up with duplicates. Cooked way of trying to sorta handle multiboxing
        fleet_list = fleet_lut.get(player['player_id'], [])
        best_entry = fleet_list.pop(best_idx)
        fleet_lut['player_id'] = fleet_list

        print('INFO: found fleet file', best_entry, 'for', player['steam_name'], 'at', game_time)
        player['fleet'] = best_entry

        return player

def process_match_result(
    match_data: MatchData,
    match_history: List[MatchData],
    database: Dict[str, PlayerData],
    model: PlackettLuce,
) -> Dict[str, List[PlackettLuceRating]]:
    updated_teams: Dict[str, List[PlackettLuceRating]] = {}

    winner = match_data["winning_team"]

    if not match_data['valid'] or winner not in match_data['teams']:
        print(f"ERROR: invalid report found at {match_data['time']}")
        return

    # create teams of rating objects for ranking, and store match data to DB
    for team_id, players in match_data['teams'].items():
        team_players: List[PlackettLuceRating] = []
        
        for player in players:
            player_id = player['player_id']
            steam_name = player['steam_name']

            if player_id not in database:
                print(f"Found new player: {steam_name} {player_id}")
                database[player_id] = {
                    "steam_name": steam_name,
                    "rating_data": model.rating(name=player_id),
                    "score": 0.0,
                    "player": True,
                    "games_played": 0,
                    "wins": 0,
                    "ans_games": 0,
                    "ans_wins": 0,
                    "osp_games": 0, 
                    "osp_wins": 0, 
                    "fortuna_games": 0, 
                    "fortuna_wins": 0, 
                    "history": [],
                    "teammates": {}
                }
            team_players.append(database[player_id]['rating_data'])

            # update total games played and wins
            database[player_id]['games_played'] += 1
            if team_id == winner:
                database[player_id]['wins'] += 1

            # update faction games played and wins
            assert(player['faction'] in ['ANS', 'OSP', 'FORTUNA'])
            if player['faction'] == 'ANS':
                database[player_id]['ans_games'] += 1
                if team_id == winner:
                    database[player_id]['ans_wins'] += 1
            if player['faction'] == 'OSP':
                database[player_id]['osp_games'] += 1
                if team_id == winner:
                    database[player_id]['osp_wins'] += 1
            if player['faction'] == 'FORTUNA':
                database[player_id]['fortuna_games'] += 1
                if team_id == winner:
                    database[player_id]['fortuna_wins'] += 1

            for teammate in players:
                teammate_id = teammate['player_id']
                if player_id == teammate_id:
                    continue
                db = database[player_id]['teammates']
                if teammate_id not in iter(db):
                    db[teammate_id] = {"games": 0, "wins": 0}
                db[teammate_id]["games"] += 1
                if team_id == winner:
                    db[teammate_id]["wins"] += 1
        updated_teams[team_id] = team_players 

    winner_team = updated_teams.pop(winner)
    other_team = updated_teams[next(iter(updated_teams))]

    # add synthetic players to balance team sizes
    largest_team = max(len(winner_team), len(other_team))

    if len(winner_team) < largest_team:
        average_mu = sum([p.mu for p in winner_team]) / len(winner_team)
        average_sigma = sum([p.sigma for p in winner_team]) / len(winner_team)

        for i in range(0, largest_team - len(winner_team)):
            winner_team.append(model.rating(
                name = "TEST_PLAYER",
                mu=average_mu,
                sigma=average_sigma
            ))

    if len(other_team) < largest_team:
        average_mu = sum([p.mu for p in other_team]) / len(other_team)
        average_sigma = sum([p.sigma for p in other_team]) / len(other_team)

        for i in range(0, largest_team - len(other_team)):
            other_team.append(model.rating(
                name = "TEST_PLAYER",
                mu=average_mu,
                sigma=average_sigma
            ))

    # grab dlo metrics for match before scoring
    # TODO: implement scoring based avg_dlo
    match_quality = math.exp(-(model.predict_win([winner_team, other_team])[0] - 0.5)** 2)

    match_data['avg_dlo'] = 0.0
    match_data['match_quality'] = match_quality

    # update scores based on predicted winner
    base_score_per_game = 25.0
    predictions = model.predict_win([winner_team, other_team])
    score_for_winners = max(1.0 - predictions[0], 0.7) * base_score_per_game
    score_for_losers = (0.9005363*math.exp(-(predictions[1] - -0.2118891) ** 2/(2*0.1952401 ** 2))) * base_score_per_game
    score_updates = [score_for_winners, score_for_losers]

    rated_teams = model.rate([winner_team, other_team])

    # write ranking update back to database
    for idx, team in enumerate(rated_teams):
        for player in team:
            if player.name == 'TEST_PLAYER':
                continue
            player_id = player.name
            database[player_id]['rating_data'] = player
            database[player_id]['score'] += score_updates[idx]
            database[player_id]['history'].append((
                match_data['time'],
                database[player_id]['score']
            ))

    # store match to match history
    match_history.append(match_data)

def update_histogram(
    histogram: HistogramType,
    database: Dict[str, PlayerData],
    game_index: int
) -> None:
    for player_id, data in database.items():
        if data['player']:
            # Changed to use player_id as key
            histogram[player_id][game_index] = data['score']

def render_leaderboard(database: Dict[str, PlayerData]) -> None:
    leaderboard = sorted(database.values(), 
                       key=lambda d: d['score'], 
                       reverse=True)

    leaderboard = [p for p in leaderboard if p['games_played'] > 0]

    output_path = Path('docs/index.html')
    
    template = jinja_env.get_template("leaderboard.html")
    html_content = template.render(
            leaderboard=leaderboard,
            depth=get_template_depth(output_path)
            )
    
    output_path.write_text(html_content)

    for player_id, data in database.items():
        render_player_page(player_id, data, database)
        render_rating_json(player_id, data)

def render_player_page(
    player_id: str,
    player_data: PlayerData,
    database: Dict[str, PlayerData]
) -> None:
    """Generate individual player page using template"""
    # Create plot
    plot_dir = Path('docs/player/images')
    plot_dir.mkdir(exist_ok=True, parents=True)
    plot_path = plot_dir / f'{player_id}_history.html'
    generate_dlo_plot(player_data, plot_path)
    
    # Calculate derived values
    total_games = player_data['games_played']
    losses = total_games - player_data['wins']
    win_rate = player_data['wins'] / total_games if total_games > 0 else 0
    
    ans_losses = player_data['ans_games'] - player_data['ans_wins']
    ans_win_rate = (player_data['ans_wins'] / player_data['ans_games'] 
                   if player_data['ans_games'] > 0 else 0)
    
    osp_losses = player_data['osp_games'] - player_data['osp_wins']
    osp_win_rate = (player_data['osp_wins'] / player_data['osp_games'] 
                   if player_data['osp_games'] > 0 else 0)

    fortuna_losses = player_data['fortuna_games'] - player_data['fortuna_wins']
    fortuna_win_rate = (player_data['fortuna_wins'] / player_data['fortuna_games'] 
                   if player_data['fortuna_games'] > 0 else 0)

    # Prepare template context
    context = {
        'player_id': player_id,
        'player_data': player_data,
        'losses': losses,
        'win_rate': win_rate,
        'ans_losses': ans_losses,
        'ans_win_rate': ans_win_rate,
        'osp_losses': osp_losses,
        'osp_win_rate': osp_win_rate,
        'fortuna_losses': fortuna_losses,
        'fortuna_win_rate': fortuna_win_rate,
        'best_friends': get_best_friends(player_data, database),
        'history_plot': open(plot_path).read(),
        'depth': get_template_depth(Path(f'docs/player/{player_id}.html'))
    }

    # Render template
    template = jinja_env.get_template("player.html")
    output_path = Path(f'docs/player/{player_id}.html')
    output_path.write_text(template.render(**context))

def render_rating_json(player_id: str, player_data: PlayerData) -> None:
    output_path = Path(f"docs/rating/{player_id}.json")
    
    json_template = {
        "version": 1,  # Version for schema changes
        "dlo": 0.0,
        "last_updated": None
    }
    
    rating_data = json_template.copy()
    rating_data.update({
        "dlo": player_data['score'],
        "last_updated": datetime.now().isoformat()
    })
        
    with open(output_path, 'w') as f:
        json.dump(rating_data, f, indent=2)

def get_best_friends(player_data: PlayerData, database: Dict[str, PlayerData]) -> List[Dict[str, Any]]:
    teammates = []
    for teammate_id, stats in player_data['teammates'].items():
        # Skip teammates with zero wins
        if stats['wins'] == 0:
            continue
            
        if teammate_id in database:
            win_rate = stats['wins'] / stats['games']
            teammates.append({
                'id': teammate_id,
                'name': database[teammate_id]['steam_name'],
                'games': stats['games'],
                'wins': stats['wins'],
                'win_rate': win_rate
            })
    
    # Sort by win rate (descending), then games played (descending)
    sorted_teammates = sorted(teammates, 
                            key=lambda x: (-x['wins'], -x['win_rate']))
    
    return sorted_teammates[:3]  # Return top 3 (or fewer if less available)

def plot_rank_distribution(database: Dict[str, PlayerData], output_path: Path) -> None:
    ordinals = [p['score'] for p in database.values() if p['player']]
    
    if not ordinals:
        print("No player data available for rank distribution")
        return

    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(
            x=ordinals,
            name='Players',
            nbinsx=50,
            marker_color='#2ecc71',
            opacity=0.85,
    )
        )

    mean_val = sum(ordinals) / len(ordinals)
    median_val = sorted(ordinals)[len(ordinals)//2]

    fig.add_vline(
        x=mean_val, 
        line=dict(color='#e74c3c', width=2, dash='dash'),
        annotation=dict(text=f"Mean: {mean_val:.1f}", 
                       font=dict(color='#e74c3c'))
    )
    fig.add_vline(
        x=median_val, 
        line=dict(color='#3498db', width=2, dash='dash'),
        annotation=dict(text=f"Median: {median_val:.1f}", 
                       font=dict(color='#3498db'))
    )

    fig.update_layout(
        title='Player Rank Distribution',
        xaxis_title='DLO Rating',
        yaxis_title='Player Count',
        plot_bgcolor='#2d2d2d',
        paper_bgcolor='#1a1a1a',
        font=dict(
            family='monospace',
            color='#e0e0e0',
            size=14
        ),
        title_font_size=18,
        hovermode='x unified',
        margin=dict(l=60, r=30, t=80, b=60),
        xaxis=dict(
            showgrid=True,
            gridcolor='#3a3a3a',
            tickfont=dict(color='#e0e0e0'),
            linecolor='#3a3a3a',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#3a3a3a',
            tickfont=dict(color='#e0e0e0'),
            linecolor='#3a3a3a',
            zeroline=False
        ),
        hoverlabel=dict(
            bgcolor='#333333',
            font_size=12,
            font_family='monospace'
        ),
        legend=dict(
            bgcolor='#2d2d2d',
            font=dict(color='#e0e0e0')
        )
    )

    fig.write_html(
        output_path,
        include_plotlyjs='cdn',
        full_html=True,
        config={'displayModeBar': False}  # Cleaner display
    )

def generate_dlo_plot(
    player_data: PlayerData,
    output_path: Path
) -> None:
    history = player_data['history']
    if not history:
        return

    times, ordinals = zip(*sorted(history, key=lambda x: x[0]))
    
    fig = px.line(
        x=times,
        y=ordinals,
        markers=True,
        labels={'x': 'Date', 'y': 'DLO'},
        title=f'DLO Rating History - {player_data["steam_name"]}'
    )
    
    fig.update_layout(
        plot_bgcolor='#2d2d2d',
        paper_bgcolor='#1a1a1a',
        font=dict(
            family='monospace',
            color='#e0e0e0'
        ),
        title=dict(
            font=dict(
                color='#ffffff'
            )
        ),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor='#3a3a3a',
            tickfont=dict(color='#e0e0e0'),
            tickformat='%Y-%m-%d',
            linecolor='#3a3a3a'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#3a3a3a',
            tickfont=dict(color='#e0e0e0'),
            linecolor='#3a3a3a'
        ),
        hoverlabel=dict(
            bgcolor='#333333',
            font_size=12,
            font_family='monospace',
            font_color='#e0e0e0'
        )
    )
    
    fig.update_traces(
        line=dict(color='#00ccff', width=2),
        marker=dict(size=6, color='#00ccff'),
        hovertemplate=(
            '<span style="color:#e0e0e0">'
            '<b>%{x|%Y-%m-%d}</b><br>DLO: %{y:.2f}'
            '</span><extra></extra>'
        )
    )
    
    pio.write_html(
        fig,
        file=output_path,
        full_html=False,
        include_plotlyjs='cdn',
        default_width='100%',
        default_height='400px'
    )


def get_best_friends(player_data: PlayerData, database: Dict[str, PlayerData]) -> List[Dict[str, Any]]:
    teammates = []
    for teammate_id, stats in player_data['teammates'].items():
        # Skip teammates with zero wins
        if stats['wins'] == 0:
            continue
            
        if teammate_id in database:
            win_rate = stats['wins'] / stats['games']
            teammates.append({
                'id': teammate_id,
                'name': database[teammate_id]['steam_name'],
                'games': stats['games'],
                'wins': stats['wins'],
                'win_rate': win_rate
            })
    
    # Sort by win rate (descending), then games played (descending)
    sorted_teammates = sorted(teammates, 
                            key=lambda x: (-x['wins'], -x['win_rate']))
    
    return sorted_teammates[:3]  # Return top 3 (or fewer if less available)

def render_match_history(match_history: List[MatchData]) -> None:
    template = jinja_env.get_template("match_history.html")
    output_path = Path("docs/match_history.html")

    sorted_matches = sorted(match_history, key=lambda m: m["time"], reverse=True)
    
    html_content = template.render(
        matches=sorted_matches,
        depth=get_template_depth(output_path)
    )
    output_path.write_text(html_content)

    for match in sorted_matches:
        render_match_details(match)

def render_match_details(match_data: MatchData) -> None:
    template = jinja_env.get_template("match_details.html")
    filename = match_data["time"].strftime("%Y%m%d_%H%M%S") + ".html"
    output_path = Path(f"docs/match/{filename}")
    
    html_content = template.render(
        match=match_data,
        depth=get_template_depth(output_path)
    )
    output_path.write_text(html_content)

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
            original = database[steam_id]['score']
            
            adjusted_rating = model.rating(
                name = original.name,
                mu=original.mu + adj['mu_adjustment'],
                sigma=original.sigma
            )
            
            database[steam_id]['score'] = adjusted_rating
            print(f"Adjusted {adj['steam_name']} ({steam_id}): "
                  f"μ {original.mu:.2f} → {adjusted_rating.mu:.2f} "
                  f"({adj['mu_adjustment']:+.2f}) - {adj['reason_for_adjustment']}")
        else:
            print(f"Warning: Player {adj['steam_name']} ({steam_id}) not found")

def parse_skirmish_report_datetime(filename: Path) -> datetime:
    formats = [
        "%d-%b-%Y %H-%M-%S",    # For filenames like "14-Apr-2025 22-30-01"
        "%Y-%m-%d %H:%M:%S.%f", # For strings like "2025-03-27 16:04:52.263218"
    ]
    datetime_str = filename.name.split(" - ")[-1].replace(".xml", "")

    for fmt in formats:
        try:
            return datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Failed to parse datetime: {datetime_str}")

def main() -> None:
    model: PlackettLuce = PlackettLuce(balance=False, limit_sigma=False)
    with open('season1_database.pkl', 'rb') as file:
        database: Dict[str, PlayerData] = pickle.load(file)
    # reset per-season stats
    for player_id, player_data in database.items():
        database[player_id]['score'] = 0.0
        database[player_id]["games_played"] = 0
        database[player_id]["wins"] = 0
        database[player_id]["ans_games"] = 0
        database[player_id]["ans_wins"] = 0
        database[player_id]["osp_games"] = 0
        database[player_id]["osp_wins"] = 0
        database[player_id]["fortuna_games"] = 0
        database[player_id]["fortuna_wins"] = 0
        database[player_id]["history"] = []
        database[player_id]["teammates"] = {}
    
    with open('season1_match_history.pkl', 'rb') as file:
        match_history: List[MatchData] = pickle.load(file)
    
    # store new fleet files first
    collect_fleet_files()
    generate_fleet_images()
    fleet_lut = build_fleet_lut()

    battle_reports = sorted(
        Path("/srv/steam/.steam/steam/steamapps/common/NEBULOUS Dedicated Server/Saves/SkirmishReports/").iterdir(),
        key=parse_skirmish_report_datetime)

    for index, file in enumerate(battle_reports):
        print(f"\nPROCESSING FILE: {file}")
        match_data = parse_battle_report(file, fleet_lut)
        process_match_result(match_data, match_history, database, model)

    # Apply manual adjustments
    adjustments = load_rank_adjustments()
    if adjustments:
        print("\nApplying manual adjustments:")
        # TODO: manual adjustments deprecated after season 1, no longer applies
        # apply_manual_adjustments(model, database, adjustments)
    
    plot_rank_distribution(database, Path("docs/rank_distribution.html"))
    render_leaderboard(database)
    render_match_history(match_history)

if __name__ == "__main__":
    main()
